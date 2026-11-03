import { zodResolver } from "@hookform/resolvers/zod"
import { RotateCcw, Search } from "lucide-react"
import { useForm } from "react-hook-form"
import { z } from "zod"

import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form"
import { Input } from "@/components/ui/input"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"

// Mirrors ArticleQuery in backend/app/schemas/article.py. "all" and empty
// strings are UI-only placeholders for "filter not set".
export const filtersSchema = z
  .object({
    group_type: z.enum(["all", "competitor", "regulatory"]),
    site_name: z.string(),
    date_from: z.string(),
    date_to: z.string(),
    search: z.string(),
  })
  .refine(
    (data) =>
      !data.date_from || !data.date_to || data.date_from <= data.date_to,
    {
      message: "The from date must be on or before the to date",
      path: ["date_to"],
    },
  )

export type ArticleFilters = z.infer<typeof filtersSchema>

export const emptyFilters: ArticleFilters = {
  group_type: "all",
  site_name: "",
  date_from: "",
  date_to: "",
  search: "",
}

// Only /articles/history accepts these — /articles/today filters by group only.
const historyFields = [
  {
    name: "site_name",
    label: "Site name",
    type: "text",
    placeholder: "Exact site name",
  },
  {
    name: "search",
    label: "Keyword",
    type: "text",
    placeholder: "Title or site name",
  },
  { name: "date_from", label: "Published from", type: "date" },
  { name: "date_to", label: "Published to", type: "date" },
] as const

interface ArticlesFiltersProps {
  mode: "today" | "history"
  values: ArticleFilters
  onApply: (filters: ArticleFilters) => void
}

export function ArticlesFilters({
  mode,
  values,
  onApply,
}: ArticlesFiltersProps) {
  const form = useForm<ArticleFilters>({
    resolver: zodResolver(filtersSchema),
    mode: "onBlur",
    criteriaMode: "all",
    defaultValues: values,
  })

  const handleReset = () => {
    form.reset(emptyFilters)
    onApply(emptyFilters)
  }

  return (
    <Card>
      <CardContent>
        <Form {...form}>
          <form
            onSubmit={form.handleSubmit(onApply)}
            className="flex flex-col gap-4"
          >
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
              <FormField
                control={form.control}
                name="group_type"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Group</FormLabel>
                    <Select value={field.value} onValueChange={field.onChange}>
                      <FormControl>
                        <SelectTrigger data-testid="group-type-select">
                          <SelectValue placeholder="All groups" />
                        </SelectTrigger>
                      </FormControl>
                      <SelectContent>
                        <SelectItem value="all">All groups</SelectItem>
                        <SelectItem value="competitor">Competitor</SelectItem>
                        <SelectItem value="regulatory">Regulatory</SelectItem>
                      </SelectContent>
                    </Select>
                    <FormMessage />
                  </FormItem>
                )}
              />

              {mode === "history" &&
                historyFields.map((historyField) => (
                  <FormField
                    key={historyField.name}
                    control={form.control}
                    name={historyField.name}
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>{historyField.label}</FormLabel>
                        <FormControl>
                          <Input
                            data-testid={`${historyField.name}-input`}
                            type={historyField.type}
                            placeholder={
                              "placeholder" in historyField
                                ? historyField.placeholder
                                : undefined
                            }
                            {...field}
                          />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                ))}
            </div>

            <div className="flex items-center gap-2">
              <Button type="submit">
                <Search />
                Apply filters
              </Button>
              <Button type="button" variant="outline" onClick={handleReset}>
                <RotateCcw />
                Reset
              </Button>
            </div>
          </form>
        </Form>
      </CardContent>
    </Card>
  )
}

export default ArticlesFilters
