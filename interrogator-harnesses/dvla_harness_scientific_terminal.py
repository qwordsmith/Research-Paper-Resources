import os
from agent_interrogator import AgentInterrogator, InterrogationConfig, LLMConfig, ModelProvider, OutputMode
from dotenv import load_dotenv
import aiohttp
import asyncio
from typing import Dict, Any, Optional, List, Counter as TypeCounter
from collections import defaultdict, Counter
import json
import ast
import playwright
import time

load_dotenv(override=True)

class PlaywrightCallback:
    """Callback for interacting with agents through web interfaces using Playwright.
    
    This callback allows interaction with agents that are accessible through web UIs,
    such as chat interfaces or web-based playgrounds. It supports multiple browsers
    and can handle complex UI interactions.
    """
    
    def __init__(
        self,
        url: str,
        prompt_selector: str,
        submit_selector: str,
        response_selector: str,
        browser_type: str = "chromium",
        headless: bool = True
    ):
        """Initialize the Playwright callback.
        
        Args:
            url: The URL of the web interface
            prompt_selector: CSS selector for the prompt input element
            submit_selector: CSS selector for the submit button
            response_selector: CSS selector for the response element
            browser_type: Browser to use (chromium, firefox, or webkit)
            headless: Whether to run browser in headless mode
        """
        self.url = url
        self.prompt_selector = prompt_selector
        self.submit_selector = submit_selector
        self.response_selector = response_selector
        self.browser_type = browser_type
        self.headless = headless
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
    
    async def initialize(self):
        """Initialize the browser and page."""
        from playwright.async_api import async_playwright
        
        self.playwright = await async_playwright().start()
        browser_client = getattr(self.playwright, self.browser_type)
        self.browser = await browser_client.launch(headless=self.headless)
        self.context = await self.browser.new_context()
        self.page = await self.context.new_page()
        await self.page.goto(self.url)
        
        # Wait for the page to be fully loaded
        await self.page.wait_for_load_state("networkidle")
    
    async def cleanup(self):
        """Clean up browser resources."""
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
    
    async def __call__(self, prompt: str) -> str:
        """Send prompt through web interface and get response."""
        if not self.page:
            await self.initialize()
        
        try:
            # Type the prompt
            await self.page.fill(self.prompt_selector, prompt)
            
            # Click submit and wait for response
            await self.page.click(self.submit_selector)
            
            # Wait for and get the response
            ### TODO: Introduce enhancement watching for the final response to load instead of a generic timeout 
            await self.page.wait_for_timeout(12000)
            paragraphs = self.page.locator(self.response_selector)
            response = await paragraphs.last.text_content()
            print('extracted response:')
            print(response)
            return response
            
        except Exception as e:
            await self.cleanup()
            raise RuntimeError(f"Failed to interact with web interface: {str(e)}")

# Defining Config for AgentInterrogator
config = InterrogationConfig(
    llm=LLMConfig(
        provider=ModelProvider.OPENAI,  # or ModelProvider.HUGGINGFACE
        model_name="gpt-4.1",
        api_key=os.getenv("API_KEY")
    ),
    max_iterations=5,  # Maximum cycles for capability discovery
    output_mode=OutputMode.VERBOSE
)

# Initializing Playwright Callback Instance for AgentInterrogator
callback = PlaywrightCallback(
    url="http://localhost:8501/",
    # CSS selectors for key elements
    prompt_selector='textarea[type="textarea"]',
    submit_selector='button[data-testid="stChatInputSubmitButton"]',
    response_selector='div[data-testid="stChatMessageContent"]',
    # Browser configuration
    browser_type="chromium",  # or "firefox", "webkit"
    headless=True
)

class ScientificTester:
    """Scientific tester for agent interrogation results.
    
    Runs multiple interrogations and tracks statistics about discovered functions,
    parameters, and return types.
    """
    
    def __init__(self, num_runs: int = 20):
        """Initialize the scientific tester.
        
        Args:
            num_runs: Number of interrogation runs to perform
        """
        self.num_runs = num_runs
        # Presence counters (count a maximum of 1 per run)
        self.function_counter: Counter[str] = Counter()
        self.param_counters: defaultdict[str, Counter[str]] = defaultdict(Counter)
        self.return_type_counters: defaultdict[str, Counter[str]] = defaultdict(Counter)

        # Duplicate mention counters (how many extra times an item was repeated within runs)
        self.dup_function_mentions: Counter[str] = Counter()
        self.dup_param_mentions: defaultdict[str, Counter[str]] = defaultdict(Counter)
        self.dup_return_type_mentions: defaultdict[str, Counter[str]] = defaultdict(Counter)

        # Tracks how many runs produced more than one distinct return type for a function
        self.multi_return_type_runs: Counter[str] = Counter()
        # Tracks, per return type, how many of its appearances were in runs with multiple return types
        self.multi_rt_per_type: defaultdict[str, Counter[str]] = defaultdict(Counter)

        self.total_runs_completed = 0
        
    def parse_profile(self, profile_data: Dict[str, Any]) -> None:
        """Parse a profile and update counters.
        
        Args:
            profile_data: The profile data returned by the interrogator
        """
        # Increment total runs counter
        self.total_runs_completed += 1

        # Local tallies for this run
        func_counts: Counter[str] = Counter()
        param_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
        rt_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)

        # Extract raw lists from nested profile structure
        capabilities = profile_data.get("capabilities", [])

        # Gather all functions across capabilities into a single iterable
        functions_iter: list[Any] = []
        for cap in capabilities:
            # Each capability may still be a pydantic object → convert to dict if needed
            if not isinstance(cap, dict) and hasattr(cap, "model_dump"):
                cap = cap.model_dump()
            if isinstance(cap, dict):
                functions_iter.extend(cap.get("functions", []))

        for func in functions_iter:
            # Ensure func is a plain dict
            if not isinstance(func, dict) and hasattr(func, "model_dump"):
                func = func.model_dump()
            if not isinstance(func, dict):
                continue

            fname = func.get("name", "unknown")
            func_counts[fname] += 1

            # Parameters
            for p in func.get("parameters", []):
                if not isinstance(p, dict) and hasattr(p, "model_dump"):
                    p = p.model_dump()
                key = f"{p.get('name', 'unknown')}: {p.get('type', 'unknown')}"
                param_counts[fname][key] += 1

            # Return type
            rt_key = func.get("return_type", "unknown")
            rt_counts[fname][rt_key] += 1

        # Update presence & duplicate counters
        for fname, cnt in func_counts.items():
            # Presence (at least once)
            self.function_counter[fname] += 1
            # Duplicates (extra mentions beyond the first)
            extra = max(0, cnt - 1)
            if extra:
                self.dup_function_mentions[fname] += extra

        for fname, pcounter in param_counts.items():
            for pkey, pcnt in pcounter.items():
                # Presence
                self.param_counters[fname][pkey] += 1
                # Duplicates
                extra = max(0, pcnt - 1)
                if extra:
                    self.dup_param_mentions[fname][pkey] += extra

        for fname, rtcounter in rt_counts.items():
            for rtkey, rtcnt in rtcounter.items():
                self.return_type_counters[fname][rtkey] += 1
                extra = max(0, rtcnt - 1)
                if extra:
                    self.dup_return_type_mentions[fname][rtkey] += extra

        # After gathering counts, check for multiple return types in this run
        for fname, rtcounter in rt_counts.items():
            if len(rtcounter) > 1:
                self.multi_return_type_runs[fname] += 1
                # Mark each return type as having appeared in a multi-return run
                for rtkey in rtcounter.keys():
                    self.multi_rt_per_type[fname][rtkey] += 1
    
    def calculate_percentages(self) -> Dict[str, Any]:
        """Calculate percentages for all tracked items.
        
        Returns:
            Dictionary with percentage statistics
        """
        results = {
            "total_runs": self.total_runs_completed,
            "functions": [],
        }
        
        # Calculate function percentages
        for func_name, count in self.function_counter.items():
            percentage = (count / self.total_runs_completed) * 100
            
            function_result = {
                "name": func_name,
                "occurrence_percentage": percentage,
                "occurrence_count": count,
                "parameters": [],
                "return_types": []
            }
            
            # Calculate parameter percentages (relative to function appearances)
            for param, param_count in self.param_counters[func_name].items():
                param_percentage = (param_count / count) * 100
                function_result["parameters"].append({
                    "name": param,
                    "occurrence_percentage": param_percentage,
                    "occurrence_count": param_count
                })
            
            # Calculate return type percentages (relative to function appearances)
            for return_type, rt_count in self.return_type_counters[func_name].items():
                rt_percentage = (rt_count / count) * 100
                function_result["return_types"].append({
                    "type": return_type,
                    "occurrence_percentage": rt_percentage,
                    "occurrence_count": rt_count
                })
            
            results["functions"].append(function_result)
        
        return results
    
    def print_results(self) -> None:
        """Print the results in a readable format."""
        results = self.calculate_percentages()
        
        print("\n" + "="*80)
        print(f"SCIENTIFIC TESTING RESULTS ({results['total_runs']} RUNS)")
        print("="*80)
        
        # Sort functions by occurrence percentage (descending)
        sorted_functions = sorted(
            results["functions"], 
            key=lambda x: x["occurrence_percentage"], 
            reverse=True
        )
        
        for func in sorted_functions:
            fname = func['name']
            print(f"\nFUNCTION: {fname}")
            print(f"  Presence: {func['occurrence_count']}/{results['total_runs']} runs ({func['occurrence_percentage']:.1f}%)")
            dup_fn = self.dup_function_mentions.get(fname, 0)
            if dup_fn:
                avg_dup = dup_fn / results['total_runs']
                print(f"  Duplicate mentions: {dup_fn} (avg {avg_dup:.2f} per run)")
            
            if func["parameters"]:
                print("  Parameters:")
                # Sort parameters by occurrence percentage (descending)
                sorted_params = sorted(
                    func["parameters"], 
                    key=lambda x: x["occurrence_percentage"], 
                    reverse=True
                )
                for param in sorted_params:
                    pname = param['name']
                    print(f"    - {pname}: {param['occurrence_count']}/{func['occurrence_count']} runs ({param['occurrence_percentage']:.1f}%)")
                    dup_p = self.dup_param_mentions[fname].get(pname, 0)
                    if dup_p:
                        print(f"        * duplicate mentions: {dup_p}")
            
            if func["return_types"]:
                print("  Return Types:")
                # Sort return types by occurrence percentage (descending)
                sorted_returns = sorted(
                    func["return_types"], 
                    key=lambda x: x["occurrence_percentage"], 
                    reverse=True
                )
                for rt in sorted_returns:
                    rt_type = rt['type']
                    multi_flag = " (multi)" if self.multi_rt_per_type[fname].get(rt_type, 0) else ""
                    print(f"    - {rt_type}{multi_flag}: {rt['occurrence_count']}/{func['occurrence_count']} runs ({rt['occurrence_percentage']:.1f}%)")
                    dup_rt = self.dup_return_type_mentions[fname].get(rt_type, 0)
                    if dup_rt:
                        print(f"        * duplicate mentions: {dup_rt}")

                # If multiple return types ever occurred, report how often
                multi_rt = self.multi_return_type_runs.get(fname, 0)
                if multi_rt:
                    pct_multi = (multi_rt / results['total_runs']) * 100
                    print(f"  NOTE: Multiple return types observed in {multi_rt}/{results['total_runs']} runs ({pct_multi:.1f}%).")
        
        print("\n" + "="*80)

def _convert_profile_to_dict(profile: Any) -> Dict[str, Any]:
    """Convert the returned profile object to a dictionary.

    The underlying AgentInterrogator may return different structures depending
    on configuration. We attempt several strategies to obtain a usable dict.
    """
    # Already a mapping
    if isinstance(profile, dict):
        return profile

    # Dataclass or pydantic model with built-in conversion helpers
    for attr in ("to_dict", "dict", "model_dump", "as_dict"):
        if hasattr(profile, attr):
            try:
                return getattr(profile, attr)()
            except Exception:
                pass

    # String representation -> try JSON then literal eval
    text = str(profile).strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(text)
        except Exception:
            # Last resort
            return {}


# Main async function
async def run_scientific_test(num_runs: int = 20):
    """Run scientific testing with multiple interrogations.
    
    Args:
        num_runs: Number of interrogation runs to perform
    """
    tester = ScientificTester(num_runs)
    
    print(f"\nStarting scientific testing with {num_runs} runs...")
    start_time = time.time()
    
    for i in range(num_runs):
        print(f"\n--- Run {i+1}/{num_runs} ---")
        try:
            interrogator = AgentInterrogator(config, callback)
            profile = await interrogator.interrogate()
            
            # Parse the profile data
            profile_data = _convert_profile_to_dict(profile)
            tester.parse_profile(profile_data)
            
            print(f"Run {i+1} completed successfully")
        except Exception as e:
            print(f"Error in run {i+1}: {str(e)}")
    
    # Calculate and print results
    elapsed_time = time.time() - start_time
    print(f"\nAll runs completed in {elapsed_time:.2f} seconds")
    tester.print_results()
    
    return tester.calculate_percentages()

# Run it
if __name__ == "__main__":
    # Default to 20 runs, but allow command-line override
    import sys
    num_runs = 20
    if len(sys.argv) > 1:
        try:
            num_runs = int(sys.argv[1])
        except ValueError:
            print(f"Invalid number of runs: {sys.argv[1]}. Using default: 20")
    
    asyncio.run(run_scientific_test(num_runs))
