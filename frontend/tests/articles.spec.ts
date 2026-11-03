import { expect, test } from "@playwright/test"

test("Articles page is accessible and shows correct title", async ({
  page,
}) => {
  await page.goto("/articles")

  await expect(page.getByRole("heading", { name: "Articles" })).toBeVisible()
  await expect(
    page.getByText("Browse crawled competitor and regulatory articles"),
  ).toBeVisible()
})

test("Today view is the default and only filters by group", async ({
  page,
}) => {
  await page.goto("/articles")

  await expect(page.getByRole("tab", { name: "Today" })).toHaveAttribute(
    "data-state",
    "active",
  )
  await expect(page.getByTestId("group-type-select")).toBeVisible()
  await expect(page.getByTestId("search-input")).toBeHidden()
})

test("Articles table renders its columns", async ({ page }) => {
  await page.goto("/articles")

  await expect(page.getByRole("columnheader", { name: "Title" })).toBeVisible()
  await expect(page.getByRole("columnheader", { name: "Site" })).toBeVisible()
  await expect(
    page.getByRole("columnheader", { name: "Published" }),
  ).toBeVisible()
})

test("Switching to History exposes the history filters", async ({ page }) => {
  await page.goto("/articles")

  await page.getByRole("tab", { name: "History" }).click()

  await expect(page).toHaveURL(/view=history/)
  await expect(page.getByTestId("search-input")).toBeVisible()
  await expect(page.getByTestId("site_name-input")).toBeVisible()
  await expect(page.getByTestId("date_from-input")).toBeVisible()
})

test("Applying history filters puts them in the URL", async ({ page }) => {
  await page.goto("/articles?view=history")

  await page.getByTestId("search-input").fill("regulation")
  await page.getByTestId("site_name-input").fill("Example Site")
  await page.getByRole("button", { name: "Apply filters" }).click()

  await expect(page).toHaveURL(/search=regulation/)
  await expect(page).toHaveURL(
    /site_name=Example\+Site|site_name=Example%20Site/,
  )
})

test("Reset clears the applied history filters", async ({ page }) => {
  await page.goto("/articles?view=history&search=regulation")

  await page.getByRole("button", { name: "Reset" }).click()

  await expect(page).not.toHaveURL(/search=regulation/)
  await expect(page.getByTestId("search-input")).toHaveValue("")
})

test("An inverted publish date range is rejected", async ({ page }) => {
  await page.goto("/articles?view=history")

  await page.getByTestId("date_from-input").fill("2026-03-01")
  await page.getByTestId("date_to-input").fill("2026-01-01")
  await page.getByRole("button", { name: "Apply filters" }).click()

  await expect(
    page.getByText("The from date must be on or before the to date"),
  ).toBeVisible()
  await expect(page).not.toHaveURL(/date_from=2026-03-01/)
})

test("A non-numeric article id shows the not found page", async ({ page }) => {
  await page.goto("/articles/not-a-number")

  await expect(page.getByTestId("not-found")).toBeVisible()
})

test("Articles is reachable from the sidebar", async ({ page }) => {
  await page.goto("/")

  await page.getByRole("link", { name: "Articles" }).click()

  await expect(page).toHaveURL(/\/articles/)
  await expect(page.getByRole("heading", { name: "Articles" })).toBeVisible()
})
