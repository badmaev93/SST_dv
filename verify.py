import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        await page.goto("http://localhost:8080/")

        # Wait for the compare grid to load
        await page.wait_for_selector(".cmp-cell")

        # Get the first cell
        first_cell = await page.locator(".cmp-cell").first

        # Check attributes
        role = await first_cell.get_attribute("role")
        tabindex = await first_cell.get_attribute("tabindex")
        aria_label = await first_cell.get_attribute("aria-label")

        print(f"Role: {role}")
        print(f"TabIndex: {tabindex}")
        print(f"Aria Label: {aria_label}")

        assert role == "button"
        assert tabindex == "0"
        assert aria_label and "Выбрать период" in aria_label

        # Verify focus
        await page.keyboard.press("Tab")
        # Ensure we actually focus on the cell or something else

        # Check map loading wrapper
        spinner_wrap = await page.locator("#_map_spinner_wrap").first
        if await spinner_wrap.count() > 0:
            spinner_role = await spinner_wrap.get_attribute("role")
            spinner_label = await spinner_wrap.get_attribute("aria-label")
            print(f"Spinner Role: {spinner_role}")
            print(f"Spinner Aria Label: {spinner_label}")
            assert spinner_role == "status"
            assert spinner_label == "Загрузка карты"

        print("UI Verification Passed!")
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
