import {
  flexRender,
  getCoreRowModel,
  useReactTable,
} from "@tanstack/react-table"
import {
  ChevronLeft,
  ChevronRight,
  ChevronsLeft,
  ChevronsRight,
} from "lucide-react"

import type { ArticleRead } from "@/client"
import { Button } from "@/components/ui/button"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { columns } from "./columns"

interface ArticlesPagination {
  total: number
  skip: number
  limit: number
  /** 1-based page number. */
  onPageChange: (page: number) => void
}

interface ArticlesTableProps {
  articles: ArticleRead[]
  /** Omitted for /articles/today, which returns a plain unpaginated list. */
  pagination?: ArticlesPagination
}

function PaginationFooter({
  total,
  skip,
  limit,
  onPageChange,
}: ArticlesPagination) {
  const page = Math.floor(skip / limit) + 1
  const pageCount = Math.max(1, Math.ceil(total / limit))
  const hasPrevious = page > 1
  const hasNext = page < pageCount

  return (
    <div className="flex flex-col items-start justify-between gap-4 border-t bg-muted/20 p-4 sm:flex-row sm:items-center">
      <div className="text-sm text-muted-foreground">
        Showing {total === 0 ? 0 : skip + 1} to {Math.min(skip + limit, total)}{" "}
        of <span className="font-medium text-foreground">{total}</span> articles
      </div>

      <div className="flex items-center gap-x-6">
        <div className="flex items-center gap-x-1 text-sm text-muted-foreground">
          <span>Page</span>
          <span className="font-medium text-foreground">{page}</span>
          <span>of</span>
          <span className="font-medium text-foreground">{pageCount}</span>
        </div>

        <div className="flex items-center gap-x-1">
          <Button
            variant="outline"
            size="sm"
            className="h-8 w-8 p-0"
            onClick={() => onPageChange(1)}
            disabled={!hasPrevious}
          >
            <span className="sr-only">Go to first page</span>
            <ChevronsLeft className="h-4 w-4" />
          </Button>
          <Button
            variant="outline"
            size="sm"
            className="h-8 w-8 p-0"
            onClick={() => onPageChange(page - 1)}
            disabled={!hasPrevious}
          >
            <span className="sr-only">Go to previous page</span>
            <ChevronLeft className="h-4 w-4" />
          </Button>
          <Button
            variant="outline"
            size="sm"
            className="h-8 w-8 p-0"
            onClick={() => onPageChange(page + 1)}
            disabled={!hasNext}
          >
            <span className="sr-only">Go to next page</span>
            <ChevronRight className="h-4 w-4" />
          </Button>
          <Button
            variant="outline"
            size="sm"
            className="h-8 w-8 p-0"
            onClick={() => onPageChange(pageCount)}
            disabled={!hasNext}
          >
            <span className="sr-only">Go to last page</span>
            <ChevronsRight className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </div>
  )
}

export function ArticlesTable({ articles, pagination }: ArticlesTableProps) {
  const table = useReactTable({
    data: articles,
    columns,
    getCoreRowModel: getCoreRowModel(),
    manualPagination: true,
  })

  return (
    <div className="flex flex-col gap-4">
      <Table>
        <TableHeader>
          {table.getHeaderGroups().map((headerGroup) => (
            <TableRow key={headerGroup.id} className="hover:bg-transparent">
              {headerGroup.headers.map((header) => (
                <TableHead key={header.id}>
                  {header.isPlaceholder
                    ? null
                    : flexRender(
                        header.column.columnDef.header,
                        header.getContext(),
                      )}
                </TableHead>
              ))}
            </TableRow>
          ))}
        </TableHeader>
        <TableBody>
          {table.getRowModel().rows.length ? (
            table.getRowModel().rows.map((row) => (
              <TableRow key={row.id}>
                {row.getVisibleCells().map((cell) => (
                  <TableCell key={cell.id}>
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </TableCell>
                ))}
              </TableRow>
            ))
          ) : (
            <TableRow className="hover:bg-transparent">
              <TableCell
                colSpan={columns.length}
                className="h-32 text-center text-muted-foreground"
              >
                No articles found.
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>

      {pagination && <PaginationFooter {...pagination} />}
    </div>
  )
}

export default ArticlesTable
