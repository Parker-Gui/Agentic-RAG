import { useSuspenseQuery } from "@tanstack/react-query"
import {
  createFileRoute,
  notFound,
  Link as RouterLink,
} from "@tanstack/react-router"
import { ArrowLeft, ExternalLink } from "lucide-react"
import { Suspense } from "react"

import { ArticlesService } from "@/client"
import PendingArticle from "@/components/Pending/PendingArticle"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

function getArticleQueryOptions(articleId: number) {
  return {
    queryFn: () => ArticlesService.getArticle({ articleId }),
    queryKey: ["articles", "detail", articleId],
  }
}

export const Route = createFileRoute("/_layout/articles/$articleId")({
  component: ArticleDetail,
  loader: ({ params }) => {
    if (!/^\d+$/.test(params.articleId)) {
      throw notFound()
    }
  },
  head: () => ({
    meta: [
      {
        title: "Article - FastAPI Template",
      },
    ],
  }),
})

function ArticleContent({ articleId }: { articleId: number }) {
  const { data: article } = useSuspenseQuery(getArticleQueryOptions(articleId))

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-3">
        <div className="flex flex-wrap items-center gap-1.5">
          <Badge
            variant={
              article.group_type === "regulatory" ? "default" : "secondary"
            }
          >
            {article.group_type}
          </Badge>
          {article.is_external && <Badge variant="outline">External</Badge>}
          <Badge variant={article.embedding_done ? "outline" : "secondary"}>
            {article.embedding_done ? "Indexed" : "Not indexed"}
          </Badge>
        </div>
        <h1 className="text-2xl font-bold tracking-tight">{article.title}</h1>
        <p className="text-muted-foreground">
          {article.site_name} ·{" "}
          {article.publish_date || "No publish date recorded"}
        </p>
        <div>
          <Button asChild variant="outline" size="sm">
            <a href={article.url} target="_blank" rel="noreferrer">
              <ExternalLink />
              Open original
            </a>
          </Button>
        </div>
      </div>

      {article.summary && (
        <Card>
          <CardHeader>
            <CardTitle>Summary</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm leading-relaxed">{article.summary}</p>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader>
          <CardTitle>Full text</CardTitle>
        </CardHeader>
        <CardContent>
          {article.body_text ? (
            <p className="whitespace-pre-wrap text-sm leading-relaxed">
              {article.body_text}
            </p>
          ) : (
            <p className="text-sm italic text-muted-foreground">
              No body text was stored for this article.
            </p>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

function ArticleDetail() {
  const { articleId } = Route.useParams()

  return (
    <div className="flex flex-col gap-6">
      <div>
        <Button asChild variant="ghost" size="sm" className="-ml-2">
          <RouterLink to="/articles">
            <ArrowLeft />
            Back to articles
          </RouterLink>
        </Button>
      </div>

      <Suspense fallback={<PendingArticle />}>
        <ArticleContent articleId={Number(articleId)} />
      </Suspense>
    </div>
  )
}
