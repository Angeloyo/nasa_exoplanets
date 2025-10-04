"use client"

import { Pie, PieChart } from "recharts"

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  ChartConfig,
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart"

interface PredictionDistributionChartProps {
  exoplanets: number;
  candidates: number;
  falsePositives: number;
}

const chartConfig = {
  exoplanet: {
    label: "Exoplanet",
    color: "var(--chart-1)",
  },
  candidate: {
    label: "Candidate",
    color: "var(--chart-2)",
  },
  none: {
    label: "False Positive",
    color: "var(--chart-3)",
  },
} satisfies ChartConfig

export function PredictionDistributionChart({ exoplanets, candidates, falsePositives }: PredictionDistributionChartProps) {
  const chartData = [
    { category: "exoplanet", count: exoplanets, fill: "var(--color-exoplanet)" },
    { category: "candidate", count: candidates, fill: "var(--color-candidate)" },
    { category: "none", count: falsePositives, fill: "var(--color-none)" },
  ]

  return (
    <Card className="flex flex-col border-0 shadow-none">
      <CardHeader className="items-center pb-0">
        <CardTitle className="text-xl">Prediction Distribution</CardTitle>
        <CardDescription>Classification breakdown</CardDescription>
      </CardHeader>
      <CardContent className="flex-1 pb-0">
        <ChartContainer
          config={chartConfig}
          className="mx-auto aspect-square max-h-[300px]"
        >
          <PieChart>
            <ChartTooltip
              cursor={false}
              content={<ChartTooltipContent nameKey="category" />}
            />
            <Pie data={chartData} dataKey="count" nameKey="category" />
            <ChartLegend
              content={<ChartLegendContent nameKey="category" />}
              className="-translate-y-2 flex-wrap gap-2 [&>*]:justify-center [&>*]:whitespace-nowrap"
            />
          </PieChart>
        </ChartContainer>
      </CardContent>
    </Card>
  )
}

