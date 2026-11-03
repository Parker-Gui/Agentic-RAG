import { useSuspenseQuery } from "@tanstack/react-query"
import { createFileRoute, useNavigate } from "@tanstack/react-router"
import { Suspense } from "react"
import { z } from "zod"

import { ArticlesService } from "@/client"
import {
  type ArticleFilters,
  ArticlesFilters,
} from "@/components/Articles/ArticlesFilters"
import { ArticlesTable } from "@/components/Articles/ArticlesTable"
import PendingArticles from "@/components/Pending/PendingArticles"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"

const DEFAULT_LIMIT = 50

// Every param is optional so /articles is linkable without search params and
// the URL only ever carries the filters the user actually set.
const searchSchema = z.object({
  view: z.enum(["today", "history"]).optional().catch(undefined),
  group_type: z
    .enum(["all", "competitor", "regulatory"])
    .optional()
    .catch(undefined),
  site_name: z.string().optional().catch(undefined),
  date_from: z.string().optional().catch(undefined),
  date_to: z.string().optional().catch(undefined),
  search: z.string().optional().catch(undefined),
  page: z.coerce.number().int().min(1).optional().catch(undefined),
  limit: z.coerce.number().int().min(1).max(200).optional().catch(undefined),
})

type ArticlesSearch = z.infer<typeof searchSchema>

interface ResolvedSearch {
  view: "today" | "history"
  filters: ArticleFilters
  page: number
  limit: number
}

function resolveSearch(search: ArticlesSearch): ResolvedSearch {
  return {
    view: search.view ?? "today",
    filters: {
      group_type: search.group_type ?? "all",
      site_name: search.site_name ?? "",
      date_from: search.date_from ?? "",
      date_to: search.date_to ?? "",
      search: search.search ?? "",
    },
    page: search.page ?? 1,
    limit: search.limit ?? DEFAULT_LIMIT,
  }
}

/** "all" and "" are UI placeholders for "not set" — the API wants them absent. */
const orUndefined = (value: string) =>
  value === "" || value === "all" ? undefined : value

function getTodayQueryOptions(groupType: ArticleFilters["group_type"]) {
  return {
    queryFn: () =>
      ArticlesService.getToday({ groupType: orUndefined(groupType) }),
    queryKey: ["articles", "today", groupType],
  }
}

function getHistoryQueryOptions({ filters, page, limit }: ResolvedSearch) {
  const params = {
    groupType: orUndefined(filters.group_type),
    siteName: orUndefined(filters.site_name),
    dateFrom: orUndefined(filters.date_from),
    dateTo: orUndefined(filters.date_to),
    search: orUndefined(filters.search),
    skip: (page - 1) * limit,
    limit,
  }

  return {
    queryFn: () => ArticlesService.getHistory(params),
    queryKey: ["articles", "history", params],
  }
}

export const Route = createFileRoute("/_layout/articles/")({
  component: Articles,
  validateSearch: searchSchema,
  head: () => ({
    meta: [
      {
        title: "Articles - FastAPI Template",
      },
    ],
  }),
})

function TodayArticles({
  groupType,
}: {
  groupType: ArticleFilters["group_type"]
}) {
  const { data: articles } = useSuspenseQuery(getTodayQueryOptions(groupType))

  return <ArticlesTable articles={articles} />
}

function HistoryArticles({
  resolved,
  onPageChange,
}: {
  resolved: ResolvedSearch
  onPageChange: (page: number) => void
}) {
  const { data } = useSuspenseQuery(getHistoryQueryOptions(resolved))

  return (
    <ArticlesTable
      articles={data.items}
      pagination={{
        total: data.total,
        skip: data.skip,
        limit: data.limit,
        onPageChange,
      }}
    />
  )
}

function Articles() {
  const search = Route.useSearch()
  const navigate = useNavigate({ from: Route.fullPath })
  const resolved = resolveSearch(search)
  const { view, filters } = resolved

  const applyFilters = (next: ArticleFilters) => {
    navigate({
      search: (prev) => ({
        ...prev,
        group_type: next.group_type === "all" ? undefined : next.group_type,
        site_name: next.site_name || undefined,
        date_from: next.date_from || undefined,
        date_to: next.date_to || undefined,
        search: next.search || undefined,
        page: undefined,
      }),
    })
  }

  const changeView = (nextView: string) => {
    navigate({
      search: (prev) => ({
        ...prev,
        view: nextView === "history" ? "history" : undefined,
        page: undefined,
      }),
    })
  }

  const changePage = (page: number) => {
    navigate({ search: (prev) => ({ ...prev, page }) })
  }

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Articles</h1>
        <p className="text-muted-foreground">
          Browse crawled competitor and regulatory articles
        </p>
      </div>

      <Tabs value={view} onValueChange={changeView}>
        <TabsList>
          <TabsTrigger value="today">Today</TabsTrigger>
          <TabsTrigger value="history">History</TabsTrigger>
        </TabsList>
      </Tabs>

      {/* Remount so the form picks up filters changed outside it (back/forward, Reset). */}
      <ArticlesFilters
        key={Object.values(filters).join("|")}
        mode={view}
        values={filters}
        onApply={applyFilters}
      />

      <Suspense fallback={<PendingArticles />}>
        {view === "today" ? (
          <TodayArticles groupType={filters.group_type} />
        ) : (
          <HistoryArticles resolved={resolved} onPageChange={changePage} />
        )}
      </Suspense>
    </div>
  )
}
