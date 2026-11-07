import type { Family } from '../../data/catalog';
import { readPlotColors } from './canvas';
import { CARD_MOTION_MS, createCardMotion, motionSeed } from './cardMotion';

const isPlainClick = (event: MouseEvent) =>
	event.button === 0 && !event.metaKey && !event.ctrlKey && !event.shiftKey && !event.altKey;

function play(card: HTMLAnchorElement, preview: HTMLElement): boolean {
	const canvas = document.createElement('canvas');
	const ctx = canvas.getContext('2d');
	const { width, height } = preview.getBoundingClientRect();
	if (!ctx || width === 0 || height === 0) return false;

	const ratio = window.devicePixelRatio || 1;
	canvas.width = Math.round(width * ratio);
	canvas.height = Math.round(height * ratio);
	canvas.className = 'card-motion';
	canvas.setAttribute('aria-hidden', 'true');
	ctx.scale(ratio, ratio);
	preview.append(canvas);
	card.dataset.playing = '';

	const colors = readPlotColors(preview);
	const motion = createCardMotion(card.dataset.motion as Family, motionSeed(card.href), width / height);
	const started = performance.now();
	const frame = (now: number) => {
		const t = (now - started) / CARD_MOTION_MS;
		motion(ctx, width, height, t, colors);
		if (t < 1 && canvas.isConnected) requestAnimationFrame(frame);
	};
	requestAnimationFrame(frame);
	return true;
}

function prefetch(href: string) {
	const link = document.createElement('link');
	link.rel = 'prefetch';
	link.href = href;
	document.head.append(link);
}

document.addEventListener('click', (event) => {
	if (event.defaultPrevented || !isPlainClick(event)) return;
	const card = (event.target as Element).closest<HTMLAnchorElement>('a[data-motion]');
	const preview = card?.querySelector<HTMLElement>('[data-card-preview]');
	if (!card || !preview) return;
	// A second click means the visitor doesn't want to wait, so the link just follows.
	if ('playing' in card.dataset) return;
	if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
	if (!play(card, preview)) return;

	event.preventDefault();
	prefetch(card.href);
	// A timer, not the last frame: animation frames stop in a background tab.
	setTimeout(() => window.location.assign(card.href), CARD_MOTION_MS);
});

// Coming back with the Back button restores the page as it was left, mid-animation.
window.addEventListener('pageshow', () => {
	for (const canvas of document.querySelectorAll('.card-motion')) canvas.remove();
	for (const card of document.querySelectorAll<HTMLElement>('a[data-playing]')) delete card.dataset.playing;
});
