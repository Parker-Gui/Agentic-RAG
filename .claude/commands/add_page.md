---
description: Scaffold a new frontend page for an entity — either a create form or a read-only list + detail pair
---

Create a new frontend page for: $ARGUMENTS

Stack: Vite + TanStack Router (file-based) + TanStack Query + react-hook-form + zod + shadcn/ui + biome.

0. **Pick the branch first** — read `backend/app/schemas/<entity>.py` *and* `backend/app/api/routes/v1/<entities>.py` before writing anything:
   - `<Entity>Create` exists **and** the router exposes `POST` → **branch A: create page** (step 3A)
   - only `<Entity>Read` / `<Entity>Detail` / `<Entity>List` / `<Entity>Query` and `GET` routes → **branch B: read-only list + detail pages** (step 3B)
   - both exist → do branch B first (the list page is where the "New" button lives), then branch A
   - Never invent a writable field or a form for an endpoint the backend doesn't have. If the schema and the router disagree, the router wins.
   - Reference implementations: branch A → `frontend/src/components/Admin/AddUser.tsx` + `frontend/src/routes/_layout/admin.tsx`; branch B → `frontend/src/components/Articles/*` + `frontend/src/routes/_layout/articles.index.tsx` + `articles.$articleId.tsx`

1. **API client** (`frontend/src/client/`) — generated, never hand-written:
   - Full regeneration from this backend: `./scripts/generate-client.sh` from the repo root — it dumps `app.main.app.openapi()` to `frontend/openapi.json` (gitignored), runs `bun run generate-client`, then `bun run lint`
   - `cd frontend && bun run generate-client` alone only re-runs codegen over the `openapi.json` already on disk (config: `frontend/openapi-ts.config.ts`)
   - Current surface: `ArticlesService`, `UsersService`, `RagService`, `UtilsService`. There is **no** `ItemsService`, `LoginService` or `PrivateService` — login / recover / reset live on `UsersService` (`loginAccessToken`, `recoverPassword`, `resetPassword`)
   - Import from `@/client`: `import { type ArticleRead, ArticlesService } from "@/client"`
   - Naming: service `<Entity>sService`; method names are the operation id minus the service prefix, lower-camel (`getToday`, `getHistory`, `getArticle`, `createUser`)
   - Query/path params arrive **camelCase** even though the backend schema is snake_case: `group_type` → `groupType`, `site_name` → `siteName`, `date_from` → `dateFrom`. Request *bodies* keep the backend's snake_case field names
   - Never add `src/types/*.ts` or a `src/lib/<entity>.ts` API wrapper — `client/types.gen.ts` + `client/sdk.gen.ts` are the only source of truth
   - `src/client/**`, `src/components/ui/**`, `src/routeTree.gen.ts` are biome-ignored — never format or hand-edit them

2. **Field mapping** — mirror the backend schema exactly:
   - Use the backend names verbatim — `User` is `full_name`, `is_active`, `role`, `is_app_admin` (not `name` / `active` / `admin` / `is_superuser`)
   - `*List` responses are `{ items, total }` (paginated list endpoints also echo `{ skip, limit }`) — never `{ data, count }`
   - Enum values must match the backend enum / literal set character-for-character (`UserRole` in `backend/app/db/models/user.py`, `VALID_GROUP_TYPES` in `backend/app/schemas/article.py`)
   - If `client/types.gen.ts` disagrees with the backend schema, the checked-in client is stale — regenerate (step 1) before writing any component against it
   - Controls: `str` → `Input`, password → `PasswordInput` (`@/components/ui/password-input`), `bool` → `Checkbox`, enum → `Select`, `int` → `Input type="number"`, `date` → `Input type="date"`

3A. **Branch A — create page**

   **Form component** (`frontend/src/components/<Entities>/Add<Entity>.tsx` — PascalCase plural folder; user-facing admin screens live in `Admin/`):
   - Only `*Create` fields are writable; `*Read` fields (`id`, `created_at`, `updated_at`) never appear on a create page; `*Update`-only fields belong to the edit dialog
   - `const formSchema = z.object({...})` mirroring `<Entity>Create`, then `type FormData = z.infer<typeof formSchema>`
   - `useForm<FormData>({ resolver: zodResolver(formSchema), mode: "onBlur", criteriaMode: "all", defaultValues: {...} })` — a default for every field
   - `useMutation({ mutationFn: (data: <Entity>Create) => <Entity>sService.create<Entity>({ requestBody: data }) })`
   - `onSuccess`: `showSuccessToast("<Entity> created successfully")` + `form.reset()` + `navigate({ to: "/<entities>" })`
   - `onError: handleError.bind(showErrorToast)` — `handleError` from `@/utils`, toasts from `useCustomToast` (`@/hooks/useCustomToast`)
   - `onSettled`: `queryClient.invalidateQueries({ queryKey: ["<entities>"] })` — the same first key segment the list route uses
   - Fields wrapped in `Form` / `FormField` / `FormItem` / `FormLabel` / `FormControl` / `FormMessage` from `@/components/ui/form`
   - Required labels get `<span className="text-destructive">*</span>`; boolean fields use `<FormItem className="flex items-center gap-3 space-y-0">`
   - Submit with `<LoadingButton type="submit" loading={mutation.isPending}>` (`@/components/ui/loading-button`)
   - Page form (not a dialog): wrap in `Card` / `CardHeader` / `CardContent` instead of `Dialog*`

   **Route** (`frontend/src/routes/_layout/<entities>.new.tsx` → `/<entities>/new`):
   - `export const Route = createFileRoute("/_layout/<entities>/new")({ component: New<Entity>, head: () => ({ meta: [{ title: "New <Entity> - FastAPI Template" }] }) })`
   - Body: `grid grid-cols-1 lg:grid-cols-[2fr_1fr] gap-6` — form in the main column, guidelines in an `aside`
   - Footer actions inside the form: `Cancel` (`variant="outline"`, disabled while pending) then `LoadingButton`

   ```tsx
   function NewItem() {
     return (
       <div className="flex flex-col gap-6">
         <div>
           <h1 className="text-2xl font-bold tracking-tight">New Item</h1>
           <p className="text-muted-foreground">Create and manage your items</p>
         </div>
         <div className="grid grid-cols-1 lg:grid-cols-[2fr_1fr] gap-6">
           <AddItem />
           <aside className="rounded-xl border bg-muted/20 p-4">
             <h2 className="font-semibold mb-2">Guidelines</h2>
             <ul className="space-y-2 text-sm text-muted-foreground">
               <li>Title is required</li>
               <li>Description is optional</li>
             </ul>
           </aside>
         </div>
       </div>
     )
   }
   ```

3B. **Branch B — read-only list + detail pages**

   Three components under `frontend/src/components/<Entities>/` plus two skeletons in `frontend/src/components/Pending/`:
   - `columns.tsx` — `ColumnDef<<Entity>Read>[]`, one column per `*Read` field worth showing. The identifying column links to the detail route: `<Link to="/<entities>/$<entity>Id" params={{ <entity>Id: String(row.original.id) }}>`. Booleans render as `Badge`s, nullable fields get an explicit fallback string, external URLs use `<a target="_blank" rel="noreferrer">`
   - `<Entities>Filters.tsx` — a react-hook-form + zod form mirroring `<Entity>Query` **minus** `skip`/`limit` (those are the pagination state, not filters). Keep field names snake_case to match the backend; use `""` / `"all"` as the "unset" UI value and map to `undefined` at the call site. Cross-field constraints go in `.refine(..., { path: ["<field>"] })`. Give every control a `data-testid="<field>-input"`. Two buttons: `Apply filters` + `Reset`. Wrap in `Card` / `CardContent`. If different endpoints accept different filter subsets (e.g. `/today` takes only `group_type`), take a `mode` prop and hide the fields the active endpoint ignores
   - `<Entities>Table.tsx` — `useReactTable({ data, columns, getCoreRowModel: getCoreRowModel(), manualPagination: true })` with an explicit "No <entities> found." empty row, and an optional `pagination?: { total, skip, limit, onPageChange }` footer ("Showing X to Y of N", first / prev / next / last). **Do not** reuse `components/Common/DataTable.tsx` for a paginated endpoint — it paginates client-side over rows already fetched
   - `Pending<Entities>.tsx` / `Pending<Entity>.tsx` — `Skeleton` stand-ins with the same column count / detail blocks as the real thing

   **List route** (`frontend/src/routes/_layout/<entities>.index.tsx` → `/<entities>`):
   - `validateSearch` with a zod schema covering the filters plus `page` and `limit`. Every field must be `.optional().catch(undefined)` — a bare `.catch()` makes the param *required* in the route's types and breaks `<Link to="/<entities>">` elsewhere with TS2741. Coerce numbers: `z.coerce.number().int().min(1).optional().catch(undefined)`
   - Apply defaults in a local `resolveSearch(search)` helper, not in the schema, so a clean `/<entities>` URL stays clean
   - Write `undefined` (never `""`) in `navigate({ search: (prev) => ({ ...prev, foo: undefined }) })` to strip a param from the URL
   - Server-side pagination: `skip = (page - 1) * limit`, and the footer's `onPageChange` navigates instead of mutating local state — the URL is the single source of truth
   - One `useSuspenseQuery` per endpoint inside a small child component, wrapped by `<Suspense fallback={<Pending<Entities> />}>` in the route component. Query keys are `["<entities>", "<endpoint>", …params]` — first segment matches what any mutation invalidates
   - Multiple sibling GETs (e.g. `/today` and `/history`) become `Tabs` / `TabsList` / `TabsTrigger` driven by a `view` search param, not local state
   - Remount the filter form when the search params change from outside: `<<Entities>Filters key={Object.values(filters).join("|")} … />`

   **Detail route** (`frontend/src/routes/_layout/<entities>.$<entity>Id.tsx` → `/<entities>/$<entity>Id`):
   - Validate the path param in `loader` and `throw notFound()` on a malformed id (`if (!/^\d+$/.test(params.<entity>Id)) throw notFound()`) — otherwise a typo'd URL surfaces a backend 422 instead of the router's not-found page
   - `useSuspenseQuery` on `["<entities>", "detail", id]` inside `<Suspense fallback={<Pending<Entity> />}>`
   - Render every `*Detail` field the list omits — long text in a `Card` with `whitespace-pre-wrap`, and an italic fallback when it's empty
   - Always include a `<Link to="/<entities>">Back to <entities></Link>`

4. **Route mechanics** (both branches):
   - Flat dot-notation file names; all app pages sit directly under `routes/_layout/`. `<entities>.index.tsx` is the collection route, `<entities>.$<entity>Id.tsx` the member route
   - The `createFileRoute("...")` string must match the generated id: `/_layout/<entities>/` for the index, `/_layout/<entities>/$<entity>Id` for the member
   - Keep the `head: () => ({ meta: [{ title: "… - FastAPI Template" }] })` suffix used by sibling routes
   - Admin-only pages add `beforeLoad` with `UsersService.readUserMe()` + `throw redirect({ to: "/" })`, as in `routes/_layout/admin.tsx`
   - `routeTree.gen.ts` regenerates via the router plugin (instantly if `bun dev` is running) — never edit it
   - Auth and the app shell come from `routes/_layout.tsx`; don't re-check login or re-add padding

5. **Layout** — `_layout.tsx` already provides `p-6 md:p-8` and `mx-auto max-w-7xl`, so the page starts at `flex flex-col gap-6`:
   - Header: `h1.text-2xl.font-bold.tracking-tight` + `p.text-muted-foreground` one-liner (same as `admin.tsx` / `articles.index.tsx`)
   - List page body: header → tabs (if any) → filters card → table/pagination card
   - Detail page body: back link → badges → `h1` + subtitle line → primary action button → content cards

6. **Navigation** — make the page reachable:
   - Sidebar entry: add `{ icon, title, path }` to `baseItems` in `frontend/src/components/Sidebar/AppSidebar.tsx` (admin-only entries are appended behind the `currentUser?.is_app_admin` check)
   - From a list route to a create route: `<Button asChild><Link to="/<entities>/new">…</Link></Button>` (`Link` from `@tanstack/react-router`)

7. **Test** (`frontend/tests/<entity>.spec.ts` — flat, lowercase, e.g. `articles.spec.ts`):
   - Playwright. Branch A: heading + description visible, happy-path submit, one validation error
   - Branch B: heading + description visible; column headers via `getByRole("columnheader", …)`; the default view/tab and which filters it exposes; applying filters lands in the URL; `Reset` clears them; one cross-field validation message that does **not** navigate; a malformed detail id renders `getByTestId("not-found")`; the sidebar link reaches the page
   - Assert on URL state (`expect(page).toHaveURL(/view=history/)`) rather than on row data — the dev database contents are not fixtures
   - Helpers live in `tests/utils/` (`random.ts`, `user.ts`). `tests/utils/privateApi.ts` targets a `/private` endpoint this backend does not expose — don't build new specs on it
   - Run: `cd frontend && bun test` (needs `bunx playwright install chromium` once, plus the backend running on the port in `frontend/.env` — `uv run uvicorn app.main:app --reload --port 8000`; `--reload` is required on Windows, otherwise the Proactor event loop rejects the async DB driver)

8. Lint and typecheck: `cd frontend && bun run lint` (biome — double quotes, no semicolons, 2-space indent) then `bunx tsc -p tsconfig.build.json --noEmit`
