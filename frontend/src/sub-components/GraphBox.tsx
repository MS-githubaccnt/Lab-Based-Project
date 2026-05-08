import React, { useEffect, useRef } from 'react';
import Plotly from 'plotly.js-dist-min';
import { twMerge } from 'tailwind-merge';

interface GraphBoxProps {
  graphData: {
    data: Plotly.Data[];
    layout?: Partial<Plotly.Layout>;
  };
  className?: string;
}

const GraphBox: React.FC<GraphBoxProps> = ({ graphData, className }) => {
  const graphRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (graphRef.current && graphData) {
      const { data, layout = {} } = graphData;

      // Deep merge for xaxis and yaxis if they exist in original layout
      const newLayout = {
        ...layout,
        template: 'plotly_dark',
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: {
          ...layout.font,
          color: '#d1d5db', // zinc-300
        },
        xaxis: {
          ...layout.xaxis,
          gridcolor: '#404040', // zinc-700
          zerolinecolor: '#404040', // zinc-700
          linecolor: '#404040', // zinc-700
          tickfont: {
            ...(layout.xaxis?.tickfont || {}),
            color: '#a1a1aa' // zinc-400
          }
        },
        yaxis: {
          ...layout.yaxis,
          gridcolor: '#404040', // zinc-700
          zerolinecolor: '#404040', // zinc-700
          linecolor: '#404040', // zinc-700
            tickfont: {
            ...(layout.yaxis?.tickfont || {}),
            color: '#a1a1aa' // zinc-400
          }
        },
      };

      Plotly.newPlot(graphRef.current, data, newLayout, { displayModeBar: false });
    }
  }, [graphData]);

  return <div ref={graphRef} className={twMerge("w-full h-[400px] rounded-lg bg-zinc-900/50 border border-white/10 p-4", className)} />;
};

export default GraphBox;
