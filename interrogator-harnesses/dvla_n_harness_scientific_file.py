"""Scientific interrogation harness – clean, working version.
Runs N interrogations, aggregates statistics on functions / params / return types,
and writes a detailed report to a timestamp-named text file.
"""
from __future__ import annotations

import os
import ast
import json
import time
import asyncio
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
from typing import Any, Dict

from dotenv import load_dotenv
from agent_interrogator import (
    AgentInterrogator,
    InterrogationConfig,
    LLMConfig,
    ModelProvider,
    OutputMode,
)

load_dotenv(override=True)

################################################################################
# Playwright callback for web-UI agents                                             #
################################################################################

class PlaywrightCallback:
    """Interact with a web-based chat UI via Playwright (async)."""

    def __init__(
        self,
        url: str,
        prompt_selector: str,
        submit_selector: str,
        response_selector: str,
        browser_type: str = "chromium",
        headless: bool = True,
    ) -> None:
        self.url = url
        self.prompt_selector = prompt_selector
        self.submit_selector = submit_selector
        self.response_selector = response_selector
        self.browser_type = browser_type
        self.headless = headless
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None

    async def _ensure_page(self):
        if self._page:
            return
        from playwright.async_api import async_playwright

        self._playwright = await async_playwright().start()
        browser_factory = getattr(self._playwright, self.browser_type)
        self._browser = await browser_factory.launch(headless=self.headless)
        self._context = await self._browser.new_context()
        self._page = await self._context.new_page()
        await self._page.goto(self.url)
        await self._page.wait_for_load_state("networkidle")

    async def cleanup(self):
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    async def __call__(self, prompt: str) -> str:
        await self._ensure_page()
        try:
            await self._page.fill(self.prompt_selector, prompt)
            await self._page.click(self.submit_selector)
            # crude wait; adjust if needed
            await self._page.wait_for_timeout(10_000)
            response_nodes = self._page.locator(self.response_selector)
            return await response_nodes.last.text_content() or ""
        except Exception as exc:
            await self.cleanup()
            raise RuntimeError(f"Playwright interaction failed: {exc}")

################################################################################
# Interrogator configuration                                                       #
################################################################################

CONFIG = InterrogationConfig(
    llm=LLMConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4.1",  # adjust as desired
        api_key=os.getenv("API_KEY"),
    ),
    max_iterations=5,
    output_mode=OutputMode.VERBOSE,
)


################################################################################
# Scientific statistics helper                                                    #
################################################################################

class ScientificTester:
    """Aggregate presence/duplication statistics across multiple runs."""

    def __init__(self, num_runs: int) -> None:
        self.num_runs = num_runs
        # presence counters (max 1 per run)
        self.function_counter: Counter[str] = Counter()
        self.param_counters: defaultdict[str, Counter[str]] = defaultdict(Counter)
        self.return_type_counters: defaultdict[str, Counter[str]] = defaultdict(Counter)
        # duplication counters (extra mentions in same run)
        self.dup_function_mentions: Counter[str] = Counter()
        self.dup_param_mentions: defaultdict[str, Counter[str]] = defaultdict(Counter)
        self.dup_return_type_mentions: defaultdict[str, Counter[str]] = defaultdict(Counter)
        # multiple return-type tracking
        self.multi_return_type_runs: Counter[str] = Counter()
        self.multi_rt_per_type: defaultdict[str, Counter[str]] = defaultdict(Counter)
        self.total_runs_completed = 0

    # ---------------------------------------------------------------------
    def parse_profile(self, profile: Dict[str, Any]) -> None:
        """Update counters given a single interrogation profile (dict)."""
        self.total_runs_completed += 1

        capabilities = profile.get("capabilities", [])
        functions: list[Dict[str, Any]] = []
        for cap in capabilities:
            cap = cap.model_dump() if hasattr(cap, "model_dump") else cap
            functions.extend(cap.get("functions", []))

        seen_functions: set[str] = set()
        function_return_types_in_run: defaultdict[str, set[str]] = defaultdict(set)

        for func in functions:
            func = func.model_dump() if hasattr(func, "model_dump") else func
            fname = func.get("name") or "<unknown>"

            # presence counting (once per run)
            if fname not in seen_functions:
                self.function_counter[fname] += 1
                seen_functions.add(fname)
            else:
                # extra mention within same run
                self.dup_function_mentions[fname] += 1

            # parameters ------------------------------------------------
            seen_params: set[str] = set()
            for param in func.get("parameters", []):
                param = param.model_dump() if hasattr(param, "model_dump") else param
                pname = param.get("name")
                if not pname:
                    continue
                if pname not in seen_params:
                    self.param_counters[fname][pname] += 1
                    seen_params.add(pname)
                else:
                    self.dup_param_mentions[fname][pname] += 1

            # return types ------------------------------------------------
            seen_rtypes: set[str] = set()

            def _record_rtype(rtype_val: str) -> None:
                """Helper to normalise and record a return type appearance."""
                rtype_val = (rtype_val or "").strip()
                if not rtype_val:
                    return
                function_return_types_in_run[fname].add(rtype_val)
                if rtype_val not in seen_rtypes:
                    self.return_type_counters[fname][rtype_val] += 1
                    seen_rtypes.add(rtype_val)
                else:
                    self.dup_return_type_mentions[fname][rtype_val] += 1

            # Variant A: structured list under "return_types" → [{"type": "..."}]
            for rt_item in func.get("return_types", []):
                rt_item = rt_item.model_dump() if hasattr(rt_item, "model_dump") else rt_item
                if isinstance(rt_item, dict):
                    _record_rtype(rt_item.get("type"))
                else:
                    _record_rtype(str(rt_item))

            # Variant B: single-field return type e.g. "return_type" or "returns"
            for key in ("return_type", "returns"):
                if key in func and func[key]:
                    val = func[key]
                    if isinstance(val, (list, tuple)):
                        for v in val:
                            _record_rtype(str(v))
                    else:
                        _record_rtype(str(val))

        # post-process multiple return-type flags
        for fname, rt_set in function_return_types_in_run.items():
            if len(rt_set) > 1:
                self.multi_return_type_runs[fname] += 1
                for rt in rt_set:
                    self.multi_rt_per_type[fname][rt] += 1

    # ---------------------------------------------------------------------
    def calculate_percentages(self) -> Dict[str, Any]:
        """Return aggregated results with percentage calculations."""
        results: Dict[str, Any] = {
            "total_runs": self.total_runs_completed,
            "functions": [],
        }
        for fname, count in self.function_counter.items():
            func_result = {
                "name": fname,
                "occurrence_count": count,
                "occurrence_percentage": (count / self.total_runs_completed) * 100,
                "parameters": [],
                "return_types": [],
            }
            # parameters
            for pname, pcount in self.param_counters[fname].items():
                func_result["parameters"].append(
                    {
                        "name": pname,
                        "occurrence_count": pcount,
                        "occurrence_percentage": (pcount / count) * 100,
                    }
                )
            # return types
            for rtype, rcount in self.return_type_counters[fname].items():
                func_result["return_types"].append(
                    {
                        "type": rtype,
                        "occurrence_count": rcount,
                        "occurrence_percentage": (rcount / count) * 100,
                    }
                )
            results["functions"].append(func_result)
        return results

    # ---------------------------------------------------------------------
    def generate_report(self) -> str:
        results = self.calculate_percentages()
        lines: list[str] = ["=" * 80]
        lines.append(f"SCIENTIFIC TESTING RESULTS ({results['total_runs']} RUNS)")
        lines.append("=" * 80)

        # sort functions by discovery frequency
        for func in sorted(
            results["functions"], key=lambda f: f["occurrence_percentage"], reverse=True
        ):
            fname = func["name"]
            lines.append("")
            lines.append(f"FUNCTION: {fname}")
            lines.append(
                f"  Presence: {func['occurrence_count']}/{results['total_runs']} runs "
                f"({func['occurrence_percentage']:.1f}%)"
            )
            dup_fn = self.dup_function_mentions.get(fname, 0)
            if dup_fn:
                lines.append(f"  Duplicate mentions: {dup_fn}")

            # parameters ------------------------------------------------
            if func["parameters"]:
                lines.append("  Parameters:")
                for param in sorted(
                    func["parameters"],
                    key=lambda p: p["occurrence_percentage"],
                    reverse=True,
                ):
                    pname = param["name"]
                    lines.append(
                        f"    - {pname}: {param['occurrence_count']}/"
                        f"{func['occurrence_count']} runs "
                        f"({param['occurrence_percentage']:.1f}%)"
                    )
                    dup = self.dup_param_mentions[fname].get(pname, 0)
                    if dup:
                        lines.append(f"        * duplicate mentions: {dup}")

            # return types ----------------------------------------------
            if func["return_types"]:
                lines.append("  Return Types:")
                for rt in sorted(
                    func["return_types"],
                    key=lambda r: r["occurrence_percentage"],
                    reverse=True,
                ):
                    rtype = rt["type"]
                    multi_flag = (
                        " (multi)" if self.multi_rt_per_type[fname].get(rtype) else ""
                    )
                    lines.append(
                        f"    - {rtype}{multi_flag}: {rt['occurrence_count']}/"
                        f"{func['occurrence_count']} runs "
                        f"({rt['occurrence_percentage']:.1f}%)"
                    )
                    dup = self.dup_return_type_mentions[fname].get(rtype, 0)
                    if dup:
                        lines.append(f"        * duplicate mentions: {dup}")

                multi_runs = self.multi_return_type_runs.get(fname, 0)
                if multi_runs:
                    pct = (multi_runs / results["total_runs"]) * 100
                    lines.append(
                        f"  NOTE: Multiple return types observed in {multi_runs}/"
                        f"{results['total_runs']} runs ({pct:.1f}%)."
                    )

        # summary table -------------------------------------------------
        lines.append("\n" + "=" * 80)
        lines.append("SUMMARY TABLE (Function discovery)")
        lines.append("Function | Runs Found | % of Runs")
        lines.append("-" * 40)
        for func in sorted(
            results["functions"], key=lambda f: f["occurrence_percentage"], reverse=True
        ):
            lines.append(
                f"{func['name']} | {func['occurrence_count']} | "
                f"{func['occurrence_percentage']:.1f}%"
            )
        return "\n".join(lines)

################################################################################
# Helper
################################################################################

def _convert_profile_to_dict(profile: Any) -> Dict[str, Any]:
    """Convert AgentInterrogator profile (pydantic or otherwise) to pure dict."""
    if not profile:
        return {}
    try:
        return json.loads(profile)  # if it is JSON string
    except Exception:
        pass
    try:
        return profile.model_dump()  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        return ast.literal_eval(str(profile))  # last-ditch
    except Exception:
        return {}

################################################################################
# Runner                                                                         #
################################################################################

async def run_scientific_test(num_runs: int = 20, output_path: str | None = None):
    tester = ScientificTester(num_runs)
    print(f"\nStarting scientific testing with {num_runs} runs…")
    start = time.time()

    for i in range(num_runs):
        print(f"\n— Run {i + 1}/{num_runs} —")
        try:
            callback = PlaywrightCallback(
                url="http://localhost:8503/",
                prompt_selector="textarea[type='textarea']",
                submit_selector="button[data-testid='stChatInputSubmitButton']",
                response_selector="div[data-testid='stChatMessageContent']",
                browser_type="chromium",
                headless=True,
            )
            interrogator = AgentInterrogator(CONFIG, callback)
            profile = await interrogator.interrogate()
            tester.parse_profile(_convert_profile_to_dict(profile))
            print("Run completed ✔")
            await callback.cleanup()
        except Exception as exc:
            print(f"Run failed: {exc}")

    print(f"\nAll runs finished in {time.time() - start:.1f}s")
    report_text = tester.generate_report()

    # write to file
    out_path = (
        Path(output_path)
        if output_path
        else Path.cwd() / f"scientific_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    out_path.write_text(report_text, encoding="utf-8")
    print(f"Report written to {out_path}\n")
    return tester.calculate_percentages()

################################################################################
# CLI entry-point                                                                #
################################################################################

if __name__ == "__main__":
    import sys

    runs = 20
    if len(sys.argv) > 1:
        try:
            runs = int(sys.argv[1])
        except ValueError:
            print("Invalid run count – defaulting to 20")
    asyncio.run(run_scientific_test(runs))
