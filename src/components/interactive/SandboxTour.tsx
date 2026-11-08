import { useEffect, useReducer, useRef } from 'preact/hooks';
import { withBase } from '@/lib/url';
import { TOUR_START, TOUR_STEPS, tourReducer, type TourEvent, type TourStep } from '@/lib/viz/tour';
import Sandbox from './Sandbox';
import { buttonClass, primaryButtonClass } from './controls';

/** Remembers a finished tour in this browser only, so returning visitors see the sandbox without the steps. */
const STORAGE_KEY = 'atlas-tour-done';
const TARGET = 'data-tour-target';

function remembered(): boolean {
	try {
		return localStorage.getItem(STORAGE_KEY) === '1';
	} catch {
		return false;
	}
}

function remember(done: boolean) {
	try {
		if (done) localStorage.setItem(STORAGE_KEY, '1');
		else localStorage.removeItem(STORAGE_KEY);
	} catch {
		// Storage can be blocked. The tour still works, it just starts again next time.
	}
}

const buttonNamed = (root: Element, names: string[]) =>
	[...root.querySelectorAll('button')].filter((b) => names.includes(b.textContent?.trim() ?? ''));

/** The sandbox controls a step points at, found by role and label so the sandbox needs no tour-specific markup. */
function targetsFor(root: Element, target: TourStep['target']): Element[] {
	if (target === 'plot') return [...root.querySelectorAll('canvas[role="application"]')];
	if (target === 'run') return buttonNamed(root, ['Play', 'Pause', 'Step']);
	const timeline = root.querySelector('input[type="range"]')?.closest('label');
	return [...(timeline ? [timeline] : []), ...buttonNamed(root, ['Back'])];
}

/** Which tour action, if any, a click, key press or slider change in the sandbox amounts to. */
function actionFor(event: Event): TourEvent | null {
	const target = event.target as Element | null;
	if (!target) return null;
	if (event.type === 'input') return target.matches('input[type="range"]') ? 'rewind' : null;
	if (target.closest('canvas[role="application"]')) {
		if (event.type === 'click') return 'plot';
		const key = (event as KeyboardEvent).key;
		return key === 'Enter' || key === ' ' ? 'plot' : null;
	}
	if (event.type !== 'click') return null;
	const label = target.closest('button')?.textContent?.trim();
	if (label === 'Play' || label === 'Pause' || label === 'Step') return 'run';
	if (label === 'Back') return 'rewind';
	return null;
}

/**
 * A three-step guided tour around a real sandbox, for the home page. Each step highlights
 * a control and completes when the visitor uses it, so they learn by doing.
 */
export default function SandboxTour() {
	// The server renders the first step, so a finished tour is restored after hydration, never during it:
	// hydration keeps the server's attributes, and a different first render would leave stale classes behind.
	const [state, dispatch] = useReducer(tourReducer, TOUR_START);
	const restored = useRef(false);
	const sandbox = useRef<HTMLDivElement>(null);
	const step = TOUR_STEPS[state.step];

	useEffect(() => {
		if (remembered()) dispatch('skip');
		restored.current = true;
	}, []);

	useEffect(() => {
		if (restored.current) remember(state.done);
	}, [state.done]);

	// Listen in the capture phase, before the sandbox handles the event and re-renders.
	useEffect(() => {
		const root = sandbox.current!;
		const listen = (event: Event) => {
			const action = actionFor(event);
			if (action) dispatch(action);
		};
		for (const type of ['click', 'keydown', 'input']) root.addEventListener(type, listen, true);
		return () => {
			for (const type of ['click', 'keydown', 'input']) root.removeEventListener(type, listen, true);
		};
	}, []);

	// The sandbox loads its spec later and re-renders often, so the highlight is reapplied whenever it changes.
	useEffect(() => {
		const root = sandbox.current!;
		const highlight = () => {
			const wanted = state.done ? [] : targetsFor(root, step.target);
			for (const el of root.querySelectorAll(`[${TARGET}]`)) if (!wanted.includes(el)) el.removeAttribute(TARGET);
			for (const el of wanted) if (!el.hasAttribute(TARGET)) el.setAttribute(TARGET, '');
		};
		highlight();
		const observer = new MutationObserver(highlight);
		observer.observe(root, { childList: true, subtree: true });
		return () => {
			observer.disconnect();
			for (const el of root.querySelectorAll(`[${TARGET}]`)) el.removeAttribute(TARGET);
		};
	}, [state.step, state.done]);

	return (
		<div class="tour not-content">
			<div class="tour-panel">
				<ol class="tour-steps">
					{TOUR_STEPS.map((s, i) => {
						const finished = state.done || i < state.step;
						const current = !state.done && i === state.step;
						return (
							<li class={`tour-step ${current ? 'is-current' : ''} ${finished ? 'is-done' : ''}`} aria-current={current ? 'step' : undefined}>
								<span class="tour-badge" aria-hidden="true">{finished ? '✓' : i + 1}</span>
								<span class="tour-step-title">
									{s.title}
									{finished && <span class="sr-only"> (done)</span>}
								</span>
							</li>
						);
					})}
				</ol>
				<div aria-live="polite">
					{state.done ? (
						<p class="tour-text">
							<strong>That's all you need.</strong> Every sandbox on the site has these same controls, plus guided setups under <strong>Try this</strong>.{' '}
							<a href={withBase('/start-here/')}>Start here</a> or <a href={withBase('/catalog/')}>browse the catalog</a>.
						</p>
					) : (
						<p class="tour-text">
							<strong>
								Step {state.step + 1} of {TOUR_STEPS.length}: {step.title}.
							</strong>{' '}
							{step.text}
						</p>
					)}
				</div>
				<div class="tour-actions">
					{state.done ? (
						<button type="button" class={buttonClass} onClick={() => dispatch('restart')}>
							Replay the tour
						</button>
					) : (
						<>
							<button type="button" class={buttonClass} onClick={() => dispatch('back')} disabled={state.step === 0}>
								Back
							</button>
							<button type="button" class={primaryButtonClass} onClick={() => dispatch('next')}>
								{state.step === TOUR_STEPS.length - 1 ? 'Finish' : 'Next'}
							</button>
							<button type="button" class={`${buttonClass} tour-skip`} onClick={() => dispatch('skip')}>
								Skip the tour
							</button>
						</>
					)}
				</div>
			</div>
			<div ref={sandbox}>
				<Sandbox algorithm="knn" />
			</div>
		</div>
	);
}
