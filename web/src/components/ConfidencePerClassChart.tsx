"use client"

import { Bar, BarChart, CartesianGrid, LabelList, XAxis } from "recharts"

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
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart"

interface ConfidencePerClassChartProps {
  predictions: Array<{ prediction: string; confidence: number }>;
}

const chartConfig = {
  confidence: {
    label: "Confidence",
    color: "hsl(var(--chart-1))",
  },
} satisfies ChartConfig

export function ConfidencePerClassChart({ predictions }: ConfidencePerClassChartProps) {
  const classes = ['Exoplanet', 'Candidate', 'None'];
  
  const chartData = classes.map(cls => {
    const filtered = predictions.filter(p => p.prediction === cls);
    const avg = filtered.length > 0 
      ? filtered.reduce((sum, p) => sum + p.confidence, 0) / filtered.length 
      : 0;
    return {
      class: cls,
      confidence: Math.round(avg * 10) / 10,
    };
  });

  return (
    <Card className="border-0 shadow-none">
      <CardHeader className="items-center pb-0">
        <CardTitle className="text-xl">Confidence Per Class</CardTitle>
        <CardDescription>Average confidence by prediction type</CardDescription>
      </CardHeader>
      <CardContent>
        <ChartContainer config={chartConfig}>
          <BarChart
            accessibilityLayer
            data={chartData}
            margin={{
              top: 20,
            }}
          >
            <CartesianGrid vertical={false} />
            <XAxis
              dataKey="class"
              tickLine={false}
              tickMargin={10}
              axisLine={false}
            />
            <ChartTooltip
              cursor={false}
              content={<ChartTooltipContent hideLabel />}
            />
            <Bar dataKey="confidence" fill="hsl(var(--chart-1))" radius={8}>
              <LabelList
                position="top"
                offset={12}
                className="fill-foreground"
                fontSize={12}
              />
            </Bar>
          </BarChart>
        </ChartContainer>
      </CardContent>
    </Card>
  )
}

