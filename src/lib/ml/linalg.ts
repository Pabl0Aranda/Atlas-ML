export type Vector = number[];

export const dot = (a: Vector, b: Vector): number => a.reduce((sum, v, i) => sum + v * b[i], 0);

export const norm = (a: Vector): number => Math.sqrt(dot(a, a));

export const mean = (values: number[]): number =>
	values.length ? values.reduce((s, v) => s + v, 0) / values.length : 0;
