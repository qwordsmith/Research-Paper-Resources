import os
from agent_interrogator import AgentInterrogator, InterrogationConfig, LLMConfig, ModelProvider
from dotenv import load_dotenv
import aiohttp
import asyncio
from typing import Dict, Any, Optional
import json
import playwright

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
    max_iterations=5  # Maximum cycles for capability discovery
)

# Initializing Playwright Callback Instance for AgentInterrogator
callback = PlaywrightCallback(
    url="http://localhost:8502/",
    # CSS selectors for key elements
    prompt_selector='textarea[type="textarea"]',
    submit_selector='button[data-testid="stChatInputSubmitButton"]',
    response_selector='div[data-testid="stChatMessageContent"]',
    # Browser configuration
    browser_type="chromium",  # or "firefox", "webkit"
    headless=True
)

# Main async function
async def main():
    interrogator = AgentInterrogator(config, callback)
    profile = await interrogator.interrogate()
    print(profile)

# Run it
if __name__ == "__main__":
    asyncio.run(main())