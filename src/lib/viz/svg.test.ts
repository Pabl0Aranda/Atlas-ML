import { describe, expect, it } from 'vitest';
import { sceneToSvg } from './svg';

describe('sceneToSvg', () => {
	it('draws each class with its own shape, as the live plots do', () => {
		const svg = sceneToSvg(
			{
				points: [0, 1, 2, 3].map((cls) => ({ x: cls / 4, y: 0, cls })),
			},
			'Four classes',
		);
		expect(svg.match(/<circle/g)).toHaveLength(1);
		expect(svg.match(/<rect x=/g)).toHaveLength(1);
		// A triangle has three corners and a diamond has four.
		const polygons = [...svg.matchAll(/<polygon points="([^"]+)"/g)].map((m) => m[1].split(' ').length);
		expect(polygons).toEqual([3, 4]);
	});

	it('shapes markers, such as centroids, by class too', () => {
		const svg = sceneToSvg({ points: [], markers: [{ x: 0, y: 0, cls: 1 }] }, 'One centroid');
		expect(svg).toMatch(/<rect x="[^"]+" y="[^"]+" width="[^"]+" height="[^"]+" fill="var\(--viz-series-2\)" stroke="var\(--viz-ink\)"/);
	});
});
