import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pygame
from environments.generator import MazeGenerator
from environments.maze import Action
from agents.value_iteration import ValueIterationAgent
from agents.q_learning import QLearningAgent
from agents.sarsa_lambda import SarsaLambdaAgent
from gui.renderer import MazeRenderer

STUDENT_ID = "40305054"
MAX_EPISODE_STEPS = 500
SIDEBAR_W = 300
FOOTER_H = 84
MARGIN = 16
LEGEND_PANEL_H = 156
PLAYBACK_SPEEDS = [500, 350, 250, 150, 100, 60, 30]
DEFAULT_SPEED_INDEX = 3

COLORS = {
    "bg": (250, 250, 252),
    "sidebar": (245, 247, 250),
    "panel_border": (220, 224, 230),
    "title": (25, 35, 45),
    "text": (55, 65, 75),
    "muted": (120, 130, 140),
    "button": (52, 120, 190),
    "button_hover": (41, 98, 160),
    "button_secondary": (108, 117, 125),
    "success": (39, 174, 96),
    "warning": (230, 126, 34),
}


class Button:
    def __init__(self, rect, label, action, secondary=False):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.action = action
        self.secondary = secondary
        self.hover = False

    def draw(self, surface, font):
        base = COLORS["button_secondary"] if self.secondary else COLORS["button"]
        hover = (90, 98, 105) if self.secondary else COLORS["button_hover"]
        color = hover if self.hover else base
        pygame.draw.rect(surface, color, self.rect, border_radius=6)
        text = font.render(self.label, True, (255, 255, 255))
        surface.blit(text, text.get_rect(center=self.rect.center))

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                return self.action
        return None


class MazeApp:
    def __init__(self):
        self.running = True
        self.state = "menu"
        self.renderer = MazeRenderer(cell_size=32)
        self.gen = MazeGenerator(student_id=STUDENT_ID)
        self.base_env = None
        self.env = None
        self.agent = None
        self.visited = set()
        self.seen_states = set()
        self.buttons = []
        self.status_lines = []
        self.compare_results = []
        self.training_message = ""
        self.pending_action = None
        self.screen = None
        self.clock = None
        self.agent_state = None
        self.step_count = 0
        self.episode_pause_until = 0
        self.last_step_time = 0
        self.speed_index = DEFAULT_SPEED_INDEX
        self.maze_offset = (MARGIN, MARGIN)

    @property
    def playback_interval_ms(self):
        return PLAYBACK_SPEEDS[self.speed_index]

    def run(self):
        pygame.init()
        self.base_env = self.gen.generate()
        self._setup_window()
        self.clock = pygame.time.Clock()
        self._build_menu_buttons()

        while self.running:
            self._handle_events()
            self._update()
            self._draw()
            self.clock.tick(60)

        pygame.quit()

    def _setup_window(self):
        mw, mh = self.renderer.maze_pixel_size(self.base_env)
        width = mw + SIDEBAR_W + MARGIN * 3
        height = max(mh + FOOTER_H + MARGIN * 2 + 12, 640)
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("RL Maze Solver")
        self.renderer.atlas.ensure_loaded()
        self.maze_offset = (MARGIN, MARGIN)
        self.footer_rect = pygame.Rect(MARGIN, MARGIN + mh + 8, mw, FOOTER_H)
        self.sidebar_rect = pygame.Rect(MARGIN * 2 + mw, MARGIN, SIDEBAR_W, height - MARGIN * 2)
        self.legend_rect = pygame.Rect(
            self.sidebar_rect.x + 12,
            self.sidebar_rect.y + 118,
            SIDEBAR_W - 24,
            LEGEND_PANEL_H,
        )
        self.menu_buttons_top = self.legend_rect.bottom + 14

    def _font(self, size, bold=False):
        try:
            return pygame.font.SysFont("Segoe UI", size, bold=bold)
        except Exception:
            return pygame.font.Font(None, size)

    def _fresh_env(self):
        return self.base_env.copy()

    def _preview_env(self):
        env = self.base_env.copy()
        env.reset()
        return env

    def _sidebar_button_xw(self):
        return self.sidebar_rect.x + 16, SIDEBAR_W - 32

    def _build_menu_buttons(self):
        x, w = self._sidebar_button_xw()
        y = self.menu_buttons_top
        h = 34
        gap = 8
        items = [
            ("Manual Play", "manual"),
            ("Value Iteration", "vi"),
            ("Q-Learning", "ql"),
            ("SARSA(λ)", "sarsa"),
            ("Compare All Agents", "compare"),
            ("Generate New Maze", "new_maze"),
        ]
        self.buttons = [Button((x, y + i * (h + gap), w, h), label, action) for i, (label, action) in enumerate(items)]
        quit_y = self.sidebar_rect.bottom - 50
        self.buttons.append(Button((x, quit_y, w, 36), "Quit", "quit", secondary=True))

    def _build_playback_buttons(self):
        x, w = self._sidebar_button_xw()
        y = self.sidebar_rect.y + 250
        h = 34
        gap = 8
        self.buttons = [
            Button((x, y, w, h), "Slower  [", "speed_down", secondary=True),
            Button((x, y + h + gap, w, h), "Faster  ]", "speed_up", secondary=True),
            Button((x, self.sidebar_rect.bottom - 50, w, 36), "Back to Menu", "menu", secondary=True),
        ]

    def _build_compare_buttons(self):
        x, w = self._sidebar_button_xw()
        h = 32
        gap = 8
        row_height = 42
        results_bottom = self.sidebar_rect.y + 118 + 28 + len(self.compare_results) * row_height + 20
        y = results_bottom
        self.buttons = [
            Button((x, y, w, h), "Watch Value Iteration", "watch_vi"),
            Button((x, y + (h + gap), w, h), "Watch Q-Learning", "watch_ql"),
            Button((x, y + 2 * (h + gap), w, h), "Watch SARSA(λ)", "watch_sarsa"),
            Button((x, self.sidebar_rect.bottom - 50, w, 36), "Back to Menu", "menu", secondary=True),
        ]

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if self.state in ("manual", "agent", "compare"):
                        self._go_menu()
                    elif self.state == "menu":
                        self.running = False
                    continue

                if self.state == "agent":
                    if event.key == pygame.K_LEFTBRACKET:
                        self._change_speed(-1)
                        continue
                    if event.key == pygame.K_RIGHTBRACKET:
                        self._change_speed(1)
                        continue

                if self.state == "manual":
                    self._handle_manual_key(event.key)
                    continue

            if self.state in ("menu", "compare", "manual", "agent") and event.type in (
                pygame.MOUSEBUTTONDOWN,
                pygame.MOUSEMOTION,
            ):
                for button in self.buttons:
                    action = button.handle_event(event)
                    if action:
                        self._handle_action(action)
                        break

    def _change_speed(self, delta):
        self.speed_index = max(0, min(len(PLAYBACK_SPEEDS) - 1, self.speed_index + delta))
        if self.state == "agent":
            self._refresh_speed_status()

    def _refresh_speed_status(self):
        base = self.status_lines[:1] if self.status_lines else ["Agent playback"]
        delay = self.playback_interval_ms
        steps_per_sec = 1000 / delay
        self.status_lines = base + [
            f"Speed: {steps_per_sec:.1f} steps/s  ([ slower  ] faster)",
            "Esc = menu",
        ]

    def _handle_manual_key(self, key):
        moved = False
        if key == pygame.K_UP:
            self.env.step(Action.UP)
            moved = True
        elif key == pygame.K_DOWN:
            self.env.step(Action.DOWN)
            moved = True
        elif key == pygame.K_LEFT:
            self.env.step(Action.LEFT)
            moved = True
        elif key == pygame.K_RIGHT:
            self.env.step(Action.RIGHT)
            moved = True
        elif key == pygame.K_r:
            self.env.reset()
            self.visited = set()
            self.status_lines = ["Manual mode", "Arrow keys = move | R = reset | Esc = menu"]
            return

        if moved:
            self.visited.add(self.env.agent_pos)
            if self.env.done:
                msg = "Reached the goal!" if self.env.agent_pos == self.env.goal else "Hit a trap!"
                self.status_lines = [msg, "Press R to retry or Esc for menu."]
                self.env.reset()
                self.visited = set()

    def _handle_action(self, action):
        if action == "quit":
            self.running = False
        elif action == "menu":
            self._go_menu()
        elif action == "speed_down":
            self._change_speed(-1)
        elif action == "speed_up":
            self._change_speed(1)
        elif action == "new_maze":
            self.base_env = self.gen.generate()
            self._setup_window()
            self._build_menu_buttons()
            self.status_lines = ["New maze generated."]
        elif action == "manual":
            self.env = self._fresh_env()
            self.env.reset()
            self.visited = set()
            self.state = "manual"
            self.buttons = [self.buttons[-1]] if self.buttons else []
            back_x, back_w = self._sidebar_button_xw()
            self.buttons = [Button((back_x, self.sidebar_rect.bottom - 50, back_w, 36), "Back to Menu", "menu", secondary=True)]
            self.status_lines = ["Manual mode", "Arrow keys = move | R = reset | Esc = menu"]
        elif action in ("vi", "ql", "sarsa"):
            self.pending_action = action
            self.state = "training"
            self.buttons = []
        elif action == "compare":
            self.pending_action = "compare"
            self.state = "training"
            self.buttons = []
        elif action == "watch_vi":
            self._start_agent_playback("vi")
        elif action == "watch_ql":
            self._start_agent_playback("ql")
        elif action == "watch_sarsa":
            self._start_agent_playback("sarsa")

    def _go_menu(self):
        self.state = "menu"
        self.env = None
        self.agent = None
        self.agent_state = None
        self.visited = set()
        self.seen_states = set()
        self.compare_results = []
        self.status_lines = []
        self.speed_index = DEFAULT_SPEED_INDEX
        self._build_menu_buttons()

    def _pump_training_ui(self, message):
        self.training_message = message
        self._draw()
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def _agent_action(self, state):
        if isinstance(self.agent, (QLearningAgent, SarsaLambdaAgent)):
            return self.agent.get_action(state, greedy=True)
        return self.agent.get_action(state)

    def _evaluate_agent(self, env, agent):
        env.reset()
        state = env._get_state()
        seen = set()
        steps = 0

        while not env.done and steps < MAX_EPISODE_STEPS:
            if state in seen:
                return steps, False
            seen.add(state)

            if isinstance(agent, (QLearningAgent, SarsaLambdaAgent)):
                action = agent.get_action(state, greedy=True)
            else:
                action = agent.get_action(state)

            state, _, done, _ = env.step(action)
            steps += 1

        success = env.agent_pos == env.goal
        env.reset()
        return steps, success

    def _train_q_learning(self, episodes=3000):
        best_result = None
        for attempt in range(3):
            train_env = self._fresh_env()
            if attempt:
                self._pump_training_ui(f"Q-Learning retry {attempt + 1}/3...")
            agent = QLearningAgent(train_env)
            results = agent.train(episodes=episodes + attempt * 1000)
            agent.epsilon = 0
            steps, success = self._evaluate_agent(train_env, agent)
            avg = np.mean(results["rewards"][-50:])
            result = {
                "env": train_env,
                "agent": agent,
                "steps": steps,
                "success": success,
                "detail": f"Avg reward (last 50): {avg:.1f}",
            }
            if success:
                return result
            best_result = result
        return best_result

    def _train_agent(self, mode):
        if mode == "vi":
            self._pump_training_ui("Training Value Iteration...")
            env = self._fresh_env()
            agent = ValueIterationAgent(env)
            iters = agent.train()
            steps, success = self._evaluate_agent(env, agent)
            return {
                "mode": mode,
                "env": env,
                "agent": agent,
                "steps": steps,
                "success": success,
                "detail": f"Converged in {iters} iterations",
            }

        if mode == "ql":
            self._pump_training_ui("Training Q-Learning (3000 episodes)...")
            result = self._train_q_learning(episodes=3000)
            result["mode"] = mode
            return result

        self._pump_training_ui("Training SARSA(λ) (3000 episodes)...")
        env = self._fresh_env()
        agent = SarsaLambdaAgent(env)
        results = agent.train(episodes=3000)
        agent.epsilon = 0
        steps, success = self._evaluate_agent(env, agent)
        avg = np.mean(results["rewards"][-50:])
        return {
            "mode": mode,
            "env": env,
            "agent": agent,
            "steps": steps,
            "success": success,
            "detail": f"Avg reward (last 50): {avg:.1f}",
        }

    def _run_compare_training(self):
        results = []
        for mode, label in [("vi", "Value Iteration"), ("ql", "Q-Learning"), ("sarsa", "SARSA(λ)")]:
            if not self.running:
                return

            if mode == "vi":
                self._pump_training_ui(f"Training {label}...")
                env = self._fresh_env()
                agent = ValueIterationAgent(env)
                iters = agent.train()
                steps, success = self._evaluate_agent(env, agent)
                detail = f"{iters} iterations"
            elif mode == "ql":
                self._pump_training_ui(f"Training {label}...")
                ql_result = self._train_q_learning(episodes=3000)
                env, agent, steps, success, detail = (
                    ql_result["env"],
                    ql_result["agent"],
                    ql_result["steps"],
                    ql_result["success"],
                    ql_result["detail"],
                )
            else:
                self._pump_training_ui(f"Training {label} (3000 episodes)...")
                env = self._fresh_env()
                agent = SarsaLambdaAgent(env)
                train_results = agent.train(episodes=3000)
                agent.epsilon = 0
                steps, success = self._evaluate_agent(env, agent)
                detail = f"avg rew {np.mean(train_results['rewards'][-100:]):.1f}"

            results.append({
                "mode": mode,
                "label": label,
                "env": env,
                "agent": agent,
                "steps": steps,
                "success": success,
                "detail": detail,
            })

        self.compare_results = results
        self.state = "compare"
        self._build_compare_buttons()
        self.status_lines = ["Comparison complete — same map for all agents."]

    def _start_agent_playback(self, mode):
        match = next((r for r in self.compare_results if r["mode"] == mode), None)
        if match:
            self.env = match["env"]
            self.agent = match["agent"]
            self._begin_agent_run(match["label"])

    def _begin_agent_run(self, label):
        self.env.reset()
        self.agent_state = self.env._get_state()
        self.visited = set()
        self.seen_states = set()
        self.step_count = 0
        self.episode_pause_until = 0
        self.last_step_time = pygame.time.get_ticks()
        self.state = "agent"
        self._build_playback_buttons()
        self.status_lines = [f"Running: {label}"]
        self._refresh_speed_status()

    def _reset_agent_episode(self, message):
        self.status_lines = [message, "Restarting...", "Esc = menu"]
        self.episode_pause_until = pygame.time.get_ticks() + 1200
        self.env.reset()
        self.agent_state = self.env._get_state()
        self.visited = set()
        self.seen_states = set()
        self.step_count = 0
        self.last_step_time = pygame.time.get_ticks()

    def _update(self):
        if self.state == "training" and self.pending_action:
            action = self.pending_action
            self.pending_action = None
            if not self.running:
                return
            if action == "compare":
                self._run_compare_training()
            else:
                result = self._train_agent(action)
                labels = {"vi": "Value Iteration", "ql": "Q-Learning", "sarsa": "SARSA(λ)"}
                self.env = result["env"]
                self.agent = result["agent"]
                outcome = "Goal reached" if result["success"] else "Did not reach goal"
                self._begin_agent_run(labels[action])
                self.status_lines = [
                    f"{labels[action]} — {result['detail']}",
                    f"Path: {result['steps']} steps ({outcome})",
                ]
                self._refresh_speed_status()
            return

        if self.state != "agent":
            return

        now = pygame.time.get_ticks()
        if now < self.episode_pause_until:
            return
        if now - self.last_step_time < self.playback_interval_ms:
            return

        if self.env.done or self.step_count >= MAX_EPISODE_STEPS:
            if self.env.agent_pos == self.env.goal:
                msg = f"Reached goal in {self.step_count} steps!"
            elif self.step_count >= MAX_EPISODE_STEPS:
                msg = f"Stopped after {self.step_count} steps."
            else:
                msg = f"Episode ended after {self.step_count} steps."
            self._reset_agent_episode(msg)
            return

        if self.agent_state in self.seen_states and self.step_count > 0:
            self._reset_agent_episode("Policy loop detected — restarting.")
            return

        self.seen_states.add(self.agent_state)

        prev_pos = self.env.agent_pos
        action = self._agent_action(self.agent_state)
        self.agent_state, _, done, _ = self.env.step(action)
        self.visited.add(self.agent_state[:2])
        self.step_count += 1
        self.last_step_time = now

        if self.env.agent_pos == prev_pos and not done:
            self._reset_agent_episode("Invalid move — restarting.")

    def _draw(self):
        self.screen.fill(COLORS["bg"])
        pygame.draw.rect(self.screen, COLORS["sidebar"], self.sidebar_rect, border_radius=8)
        pygame.draw.rect(self.screen, COLORS["panel_border"], self.sidebar_rect, 1, border_radius=8)

        if self.state == "training":
            env = self._preview_env()
            visited = None
        elif self.state in ("manual", "agent"):
            env = self.env
            visited = self.visited
        else:
            env = self._preview_env()
            visited = None

        self.renderer.draw_maze(
            self.screen,
            env,
            offset=self.maze_offset,
            visited=visited,
            show_agent=self.state in ("manual", "agent"),
        )
        self.renderer.draw_status_bar(
            self.screen,
            self.footer_rect,
            self.status_lines
            or [
                f"Key: {'YES' if env.has_key else 'NO'}",
                f"Door: {'OPEN' if env.door_open else 'CLOSED'}",
            ],
        )
        self._draw_sidebar()

        if self.state == "training":
            overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            overlay.fill((255, 255, 255, 180))
            self.screen.blit(overlay, (0, 0))
            text = self._font(28, bold=True).render(self.training_message or "Training...", True, COLORS["title"])
            self.screen.blit(text, text.get_rect(center=self.screen.get_rect().center))

        pygame.display.flip()

    def _draw_sidebar(self):
        title_font = self._font(22, bold=True)
        body_font = self._font(14)
        small_font = self._font(13)
        x = self.sidebar_rect.x + 16
        y = self.sidebar_rect.y + 16

        self.screen.blit(title_font.render("RL Maze Solver", True, COLORS["title"]), (x, y))
        y += 34
        for line in (
            f"Student ID: {STUDENT_ID}",
            f"Grid: {self.base_env.rows}×{self.base_env.cols}",
            f"Wall density: {int(self.gen.wall_density * 100)}%",
        ):
            self.screen.blit(small_font.render(line, True, COLORS["muted"]), (x, y))
            y += 20

        if self.state == "menu":
            self.renderer.draw_legend_panel(self.screen, self.legend_rect)
            for button in self.buttons:
                button.draw(self.screen, body_font)

        elif self.state == "compare":
            y = self.sidebar_rect.y + 118
            self.screen.blit(body_font.render("Comparison Results", True, COLORS["title"]), (x, y))
            y += 28
            for result in self.compare_results:
                status = "OK" if result["success"] else "FAIL"
                color = COLORS["success"] if result["success"] else COLORS["warning"]
                self.screen.blit(body_font.render(f"{result['label']}: {result['steps']} steps", True, COLORS["text"]), (x, y))
                self.screen.blit(small_font.render(status, True, color), (x + 210, y + 1))
                y += 18
                self.screen.blit(small_font.render(result["detail"], True, COLORS["muted"]), (x + 8, y))
                y += 24
            for button in self.buttons:
                button.draw(self.screen, body_font)

        elif self.state in ("manual", "agent", "training"):
            y = self.sidebar_rect.y + 118
            if self.state == "manual":
                lines = ["Controls", "↑ ↓ ← →  Move", "R         Reset", "Esc       Menu"]
            elif self.state == "agent":
                lines = ["Agent playback", "Use speed buttons or [ ] keys", "Esc = menu"]
            else:
                lines = ["Please wait..."]
            for i, line in enumerate(lines):
                font = body_font if i == 0 else small_font
                color = COLORS["title"] if i == 0 else COLORS["text"]
                self.screen.blit(font.render(line, True, color), (x, y))
                y += 22
            for button in self.buttons:
                button.draw(self.screen, body_font)


if __name__ == "__main__":
    MazeApp().run()
