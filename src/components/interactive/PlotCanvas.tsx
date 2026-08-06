import type { CanvasHTMLAttributes, RefObject } from 'preact';
import { useEffect, useRef, useState } from 'preact/hooks';
import { drawGrid, drawScene, readPlotColors } from '@/lib/viz/canvas';
import type { Scene } from '@/types/viz';

type Props = CanvasHTMLAttributes<HTMLCanvasElement> & {
	scene: Scene;
	canvasRef?: RefObject<HTMLCanvasElement | null>;
};

/** Draws a scene at the size of its container, and redraws when the theme changes. */
export default function PlotCanvas({ scene, canvasRef, ...canvasProps }: Props) {
	const ownRef = useRef<HTMLCanvasElement>(null);
	const ref = canvasRef ?? ownRef;
	const [theme, setTheme] = useState(0);
	const [width, setWidth] = useState(0);
	const aspect = scene.grid ? scene.grid.rows / scene.grid.cols : 1;

	useEffect(() => {
		const resize = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width));
		resize.observe(ref.current!);
		const themes = new MutationObserver(() => setTheme((t) => t + 1));
		themes.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
		return () => {
			resize.disconnect();
			themes.disconnect();
		};
	}, []);

	useEffect(() => {
		const canvas = ref.current!;
		if (!width) return;
		const ratio = window.devicePixelRatio || 1;
		canvas.width = Math.round(width * ratio);
		canvas.height = Math.round(width * aspect * ratio);
		const ctx = canvas.getContext('2d')!;
		ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
		const colors = readPlotColors(canvas);
		if (scene.grid) drawGrid(ctx, width, scene.grid, colors);
		else drawScene(ctx, width, scene, colors);
	}, [scene, width, theme]);

	return <canvas {...canvasProps} ref={ref} style={{ aspectRatio: String(1 / aspect) }} />;
}
