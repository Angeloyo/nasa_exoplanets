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

interface CompositionDistributionChartProps {
  terrestrial: number;
  oceanWorld: number;
  iceGiant: number;
  gasGiant: number;
}

const chartConfig = {
  terrestrial: {
    label: "Terrestrial",
    color: "hsl(25, 70%, 50%)", // Brown
  },
  oceanWorld: {
    label: "Ocean World",
    color: "hsl(200, 80%, 50%)", // Blue
  },
  iceGiant: {
    label: "Ice Giant",
    color: "hsl(200, 70%, 70%)", // Light blue
  },
  gasGiant: {
    label: "Gas Giant",
    color: "hsl(0, 0%, 60%)", // Grey
  },
} satisfies ChartConfig

export function CompositionDistributionChart({ 
  terrestrial, 
  oceanWorld, 
  iceGiant, 
  gasGiant 
}: CompositionDistributionChartProps) {
  const chartData = [
    { composition: "terrestrial", count: terrestrial, fill: "hsl(25, 70%, 50%)" },
    { composition: "oceanWorld", count: oceanWorld, fill: "hsl(200, 80%, 50%)" },
    { composition: "iceGiant", count: iceGiant, fill: "hsl(200, 70%, 70%)" },
    { composition: "gasGiant", count: gasGiant, fill: "hsl(0, 0%, 60%)" },
  ].filter(item => item.count > 0) // Only show compositions that exist

  return (
    <Card className="flex flex-col border-0 shadow-none">
      <CardHeader className="items-center pb-0">
        <CardTitle className="text-xl">Composition Types</CardTitle>
        <CardDescription>Planet composition breakdown</CardDescription>
      </CardHeader>
      <CardContent className="flex-1 pb-0">
        <ChartContainer
          config={chartConfig}
          className="mx-auto aspect-square max-h-[300px]"
        >
          <PieChart>
            <ChartTooltip
              cursor={false}
              content={<ChartTooltipContent nameKey="composition" />}
            />
            <Pie data={chartData} dataKey="count" nameKey="composition" />
            <ChartLegend
              content={<ChartLegendContent nameKey="composition" />}
              className="-translate-y-2 flex-wrap gap-2 [&>*]:justify-center [&>*]:whitespace-nowrap"
            />
          </PieChart>
        </ChartContainer>
      </CardContent>
    </Card>
  )
}
