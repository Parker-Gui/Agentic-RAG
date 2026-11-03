import { Link } from "@tanstack/react-router"
import type { ColumnDef } from "@tanstack/react-table"
import { ExternalLink } from "lucide-react"

import type { ArticleRead } from "@/client"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

export const columns: ColumnDef<ArticleRead>[] = [
  {
    accessorKey: "publish_date",
    header: "Published",
    cell: ({ row }) => {
      const publishDate = row.original.publish_date
      return (
        <span
          className={cn(
            "font-mono text-xs whitespace-nowrap",
            !publishDate && "italic text-muted-foreground",
          )}
        >
          {publishDate || "Unknown"}
        </span>
      )
    },
  },
  {
    accessorKey: "site_name",
    header: "Site",
    cell: ({ row }) => (
      <span className="text-muted-foreground">{row.original.site_name}</span>
    ),
  },
  {
    accessorKey: "title",
    header: "Title",
    cell: ({ row }) => (
      <Link
        to="/articles/$articleId"
        params={{ articleId: String(row.original.id) }}
        className="block max-w-md truncate font-medium underline-offset-4 hover:underline"
      >
        {row.original.title}
      </Link>
    ),
  },
  {
    accessorKey: "group_type",
    header: "Group",
    cell: ({ row }) => (
      <Badge
        variant={
          row.original.group_type === "regulatory" ? "default" : "secondary"
        }
      >
        {row.original.group_type}
      </Badge>
    ),
  },
  {
    id: "flags",
    header: "Status",
    cell: ({ row }) => (
      <div className="flex items-center gap-1.5">
        {row.original.is_external && <Badge variant="outline">External</Badge>}
        <Badge variant={row.original.embedding_done ? "outline" : "secondary"}>
          {row.original.embedding_done ? "Indexed" : "Not indexed"}
        </Badge>
      </div>
    ),
  },
  {
    id: "actions",
    header: () => <span className="sr-only">Actions</span>,
    cell: ({ row }) => (
      <div className="flex justify-end">
        <Button asChild variant="ghost" size="icon">
          <a href={row.original.url} target="_blank" rel="noreferrer">
            <ExternalLink className="size-4" />
            <span className="sr-only">Open original article</span>
          </a>
        </Button>
      </div>
    ),
  },
]
