export interface Box {
	top: number;
	left: number;
	bottom: number;
}

export type LayoutMotion =
	| { kind: 'none' }
	| { kind: 'enter' }
	| { kind: 'exit' }
	| { kind: 'move'; dx: number; dy: number };

const MOVE_MS = 300;
const ENTER_MS = 240;
const EXIT_MS = 150;
const MAX_STAGGER_MS = 120;
const SMOOTH = 'cubic-bezier(0.2, 0.8, 0.2, 1)';

const NONE: LayoutMotion = { kind: 'none' };

/**
 * Decides what an element does when a layout changes, where a null box means
 * hidden. Only a move seen from start to end slides: one that crosses the screen
 * edge fades instead, because a long trip in a fraction of a second looks frantic.
 * Anything that stays off screen is skipped, which keeps long lists cheap.
 */
export function planMotion(before: Box | null, after: Box | null, viewportHeight: number): LayoutMotion {
	const onScreen = (box: Box | null) => !!box && box.bottom > 0 && box.top < viewportHeight;
	const wasSeen = onScreen(before);
	const isSeen = onScreen(after);
	if (wasSeen && isSeen) {
		const dx = before!.left - after!.left;
		const dy = before!.top - after!.top;
		return Math.abs(dx) < 0.5 && Math.abs(dy) < 0.5 ? NONE : { kind: 'move', dx, dy };
	}
	if (isSeen) return { kind: 'enter' };
	return wasSeen ? { kind: 'exit' } : NONE;
}

const measure = (element: HTMLElement) => (element.getClientRects().length > 0 ? element.getBoundingClientRect() : null);

/** Leaves a copy behind to fade out, because the real element is already hidden. */
function fadeOutCopy(element: HTMLElement, box: DOMRect) {
	const ghost = element.cloneNode(true) as HTMLElement;
	ghost.hidden = false;
	ghost.inert = true;
	ghost.classList.add('layout-ghost');
	ghost.setAttribute('aria-hidden', 'true');
	Object.assign(ghost.style, {
		display: '',
		top: `${box.top}px`,
		left: `${box.left}px`,
		width: `${box.width}px`,
		height: `${box.height}px`,
	});
	document.body.append(ghost);
	ghost
		.animate({ opacity: [1, 0], transform: ['scale(1)', 'scale(0.94)'] }, { duration: EXIT_MS, easing: 'ease-in' })
		.finished.finally(() => ghost.remove());
}

/**
 * Runs `change`, then animates `elements` from where they were to where they
 * landed. Measuring first means an interrupted animation carries on from the
 * position on screen, so fast typing stays smooth.
 */
export function animateLayoutChange(elements: HTMLElement[], change: () => void) {
	if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
		change();
		return;
	}

	const before = elements.map(measure);
	for (const element of elements) for (const animation of element.getAnimations()) animation.cancel();
	for (const ghost of document.querySelectorAll('.layout-ghost')) ghost.remove();

	change();

	const after = elements.map(measure);
	let entered = 0;
	elements.forEach((element, i) => {
		const motion = planMotion(before[i], after[i], window.innerHeight);
		if (motion.kind === 'move') {
			element.animate(
				{ transform: [`translate(${motion.dx}px, ${motion.dy}px)`, 'none'] },
				{ duration: MOVE_MS, easing: SMOOTH },
			);
		} else if (motion.kind === 'enter') {
			element.animate(
				{ opacity: [0, 1], transform: ['scale(0.96)', 'none'] },
				{ duration: ENTER_MS, easing: SMOOTH, delay: Math.min(entered++ * 20, MAX_STAGGER_MS), fill: 'backwards' },
			);
		} else if (motion.kind === 'exit') {
			fadeOutCopy(element, before[i]!);
		}
	});
}
