"""
Physics fidelity tests for EventSat environment improvements.

Tests P1 (multi-step compression), P2 (mode transition overhead),
P3 (3-pool data pipeline), and thermal model removal.
"""
import pytest
from src.decision_procedure.context import DecisionContext
from src.environment.scenarios.eventsat_env import EventSatEnvironment


# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------

def _make_env(**overrides):
    """Create a minimal EventSatEnvironment with optional config overrides."""
    config = {
        "step_duration_s": 60,
        "max_steps": 500,
        "anomaly_prob": 0.0,   # disable anomalies for deterministic tests
        **overrides,
    }
    env = EventSatEnvironment(config=config)
    env.reset(seed=0)
    return env


def _make_env_with_scenario(**scenario_params):
    """Create env with inline scenario_params (no YAML file)."""
    base_scenario = {
        "orbit": {"orbital_period_s": 5676, "eclipse_fraction": 0.36},
        "power": {
            "solar_panels": {"generation_sun_w": 24.0},
            "battery": {
                "capacity_wh": 84.0, "initial_soc": 0.8,
                "min_soc": 0.2, "max_soc": 1.0, "charge_efficiency": 0.9,
            },
            "consumption": {
                "charging": {"sun_w": 4.72, "eclipse_w": 4.32},
                "payload_observe": {"sun_w": 17.94, "eclipse_w": 17.55},
                "payload_compress": {"sun_w": 12.77, "eclipse_w": 12.37},
                "payload_detect": {"sun_w": 12.77, "eclipse_w": 12.37},
                "payload_send": {"sun_w": 12.77, "eclipse_w": 12.37},
                "communication": {"sun_w": 33.65, "eclipse_w": 33.24},
                "safe": {"sun_w": 9.58, "eclipse_w": 9.58},
            },
        },
        "storage": {
            "obc_capacity_mb": 512.0,
            "jetson_capacity_mb": 2048.0,
            "observation_size_mb": 2.0,
            **scenario_params.get("storage", {}),
        },
        "communications": {
            "sband": {"downlink_rate_kbps": 128},
            "passes": {"daily_downlink_budget_mb": 27.0},
        },
        "payload": {
            "compression_time_factor": 2.0,
            "detection_time_s": 300.0,
        },
        "modes": {
            "constraints": {
                "payload_observe": {"min_battery_soc": 0.4},
                "payload_compress": {"min_battery_soc": 0.3},
                "payload_detect": {"min_battery_soc": 0.3},
                "payload_send": {"min_battery_soc": 0.3},
            },
            **scenario_params.get("modes", {}),
        },
        "objectives": {
            "total_observation_hours": 2.0,
            "min_downlinked_data_mb": 240.0,
            "mission_duration_days": 90.0,
        },
    }
    config = {
        "step_duration_s": 60,
        "max_steps": 500,
        "anomaly_prob": 0.0,
        "scenario_params": base_scenario,
    }
    env = EventSatEnvironment(config=config)
    env.reset(seed=0)
    return env


def _make_env_with_lottery():
    """EventSat env with the launch lottery enabled (RAAN/ArgP/TA randomized)."""
    config = {
        "step_duration_s": 60,
        "max_steps": 100,
        "anomaly_prob": 0.0,
        "scenario_params": {
            "orbit": {
                "orbital_period_s": 5676,
                "eclipse_fraction": 0.36,
                "altitude_km": 400.0,
                "inclination_deg": 97.4,
                "launch_lottery": True,
            },
            "communications": {
                "sband": {"downlink_rate_kbps": 128},
                "passes": {"daily_downlink_budget_mb": 27.0},
            },
        },
    }
    return EventSatEnvironment(config=config)


class TestEpisodeOrbitPersistence:
    """The env exposes the actual per-episode orbit + pass schedule."""

    def test_records_lottery_draws(self):
        env = _make_env_with_lottery()
        env.reset(seed=7)
        orbit = env.get_episode_orbit()
        for k in ("raan_deg", "arg_perigee_deg", "true_anomaly_deg"):
            assert k in orbit
            assert 0.0 <= orbit[k] <= 360.0

    def test_deterministic_per_seed(self):
        env = _make_env_with_lottery()
        env.reset(seed=7)
        a = env.get_episode_orbit()["raan_deg"]
        env.reset(seed=7)
        b = env.get_episode_orbit()["raan_deg"]
        env.reset(seed=8)
        c = env.get_episode_orbit()["raan_deg"]
        assert a == b  # same seed → same orbit
        assert a != c  # different seed → different orbit

    def test_ground_passes_serializable(self):
        env = _make_env_with_lottery()
        env.reset(seed=7)
        passes = env.get_ground_passes()
        assert isinstance(passes, list)
        for gp in passes:
            assert isinstance(gp, dict)
            assert "start_step" in gp and "end_step" in gp


# -----------------------------------------------------------------
# P1: Multi-step compression pipeline
# -----------------------------------------------------------------

class TestCompressionPipeline:
    """P1: Compression takes compression_time_factor steps, not 1."""

    def test_compression_takes_multiple_steps(self):
        """With factor=2.0, one observation takes 2 compress steps to complete."""
        env = _make_env_with_scenario(storage={"compression_ratio": 1.0})
        # First observe to get raw data
        env.step({"eventsat_0": {"mode": "payload_observe"}})
        assert env.uncompressed_observations == 1
        assert env.jetson_raw_mb == pytest.approx(2.0)

        # First compress step: in progress, not yet complete
        result1 = env.step({"eventsat_0": {"mode": "payload_compress"}})
        assert env.uncompressed_observations == 1  # not yet done
        assert env.compression_progress == 1
        assert result1.info.get("compression_completed") is None or \
               result1.info.get("compression_in_progress", False)

        # Second compress step: should complete
        env.step({"eventsat_0": {"mode": "payload_compress"}})
        assert env.uncompressed_observations == 0  # done
        assert env.compression_progress == 0       # reset after completion

    def test_compression_progress_resets_on_mode_switch(self):
        """Switching away from compress resets progress (penalizes thrashing)."""
        env = _make_env_with_scenario()
        env.step({"eventsat_0": {"mode": "payload_observe"}})
        # Start compressing
        env.step({"eventsat_0": {"mode": "payload_compress"}})
        assert env.compression_progress == 1
        # Switch to charging
        env.step({"eventsat_0": {"mode": "charging"}})
        assert env.compression_progress == 0

    def test_no_compression_reward_without_data(self):
        """Compressing with no uncompressed data gives failed_action reward."""
        env = _make_env_with_scenario()
        assert env.uncompressed_observations == 0
        result = env.step({"eventsat_0": {"mode": "payload_compress"}})
        info = result.info
        assert info.get("had_data_to_compress") is False

    def test_compression_completes_moves_data_to_compressed_pool(self):
        """After compression completes, data moves from jetson_raw to jetson_compressed."""
        env = _make_env_with_scenario(storage={"compression_ratio": 2.0})
        env.step({"eventsat_0": {"mode": "payload_observe"}})
        raw_before = env.jetson_raw_mb  # 2.0 MB

        # Run 2 compress steps (factor=2.0)
        env.step({"eventsat_0": {"mode": "payload_compress"}})
        env.step({"eventsat_0": {"mode": "payload_compress"}})

        # Data moved: 2.0 MB raw → 1.0 MB compressed (ratio=2.0)
        assert env.uncompressed_observations == 0
        assert env.jetson_raw_mb < raw_before
        assert env.jetson_compressed_mb == pytest.approx(1.0, abs=0.01)


# -----------------------------------------------------------------
# P2: Mode transition overhead
# -----------------------------------------------------------------

class TestModeTransition:
    """P2: Attitude maneuvers incur settling time overhead."""

    def _make_env_with_transition(self, settling_steps=2):
        """Create env with transition overhead enabled (settling_steps * 60s)."""
        return _make_env_with_scenario(modes={
            "constraints": {
                "payload_observe": {"min_battery_soc": 0.4},
                "payload_compress": {"min_battery_soc": 0.3},
            },
            "transition_overhead": {
                "settling_time_s": settling_steps * 60.0,
                "attitude_maneuver_modes": ["payload_observe", "communication"],
            },
        })

    def test_transition_incurred_when_switching_to_observe(self):
        """charging → payload_observe requires settling: first step executes as charging."""
        env = self._make_env_with_transition(settling_steps=2)
        assert env.previous_mode == "charging"
        # Request observe — should transition first
        result = env.step({"eventsat_0": {"mode": "payload_observe"}})
        # During transition the effective mode should be charging (non-productive)
        assert result.info["resolved_mode"] == "charging"
        assert result.info["in_transition"] is True
        assert env.total_observation_s == 0.0  # no science during transition

    def test_transition_completes_after_settling_steps(self):
        """After settling_steps, observe executes normally."""
        env = self._make_env_with_transition(settling_steps=2)
        # Step 1 & 2: in transition
        env.step({"eventsat_0": {"mode": "payload_observe"}})
        env.step({"eventsat_0": {"mode": "payload_observe"}})
        # Step 3: transition complete, observe should execute
        result = env.step({"eventsat_0": {"mode": "payload_observe"}})
        assert result.info["in_transition"] is False
        assert result.info["resolved_mode"] == "payload_observe"
        assert env.total_observation_s > 0.0

    def test_no_transition_for_same_mode(self):
        """Staying in same mode incurs no overhead."""
        env = self._make_env_with_transition(settling_steps=2)
        # Get into observe mode first (skip through transitions)
        for _ in range(3):
            env.step({"eventsat_0": {"mode": "payload_observe"}})
        # Now in observe mode — next observe should have no transition
        result = env.step({"eventsat_0": {"mode": "payload_observe"}})
        assert result.info["in_transition"] is False

    def test_no_transition_for_non_maneuver_modes(self):
        """charging → payload_compress does not require attitude maneuver."""
        env = self._make_env_with_transition(settling_steps=2)
        result = env.step({"eventsat_0": {"mode": "payload_compress"}})
        assert result.info["in_transition"] is False

    def test_zero_settling_time_no_overhead(self):
        """Default config (no transition_overhead) has zero overhead."""
        env = _make_env_with_scenario()  # no transition_overhead key
        assert env.settling_time_steps == 0
        result = env.step({"eventsat_0": {"mode": "payload_observe"}})
        assert result.info["in_transition"] is False


# -----------------------------------------------------------------
# P3: 3-pool data pipeline
# -----------------------------------------------------------------

class TestDataPipeline:
    """P3: Observe→Jetson raw → Compress→Jetson compressed → Transfer→OBC → Downlink."""

    def _make_pipeline_env(self, compression_ratio=2.0, transfer_kbps=50):
        return _make_env_with_scenario(storage={
            "compression_ratio": compression_ratio,
            "jetson_to_obc_rate_kbps": transfer_kbps,
        })

    def test_observe_adds_to_jetson_raw(self):
        env = self._make_pipeline_env()
        env.step({"eventsat_0": {"mode": "payload_observe"}})
        assert env.jetson_raw_mb == pytest.approx(2.0)
        assert env.jetson_compressed_mb == 0.0
        assert env.obc_data_mb == 0.0

    def test_compress_moves_raw_to_compressed_with_ratio(self):
        """After compression, jetson_raw decreases and jetson_compressed gets compressed size."""
        env = self._make_pipeline_env(compression_ratio=4.0)
        env.step({"eventsat_0": {"mode": "payload_observe"}})
        # Run 2 compress steps (factor=2.0)
        env.step({"eventsat_0": {"mode": "payload_compress"}})
        env.step({"eventsat_0": {"mode": "payload_compress"}})
        # 2 MB / 4.0 = 0.5 MB compressed
        assert env.jetson_raw_mb == pytest.approx(0.0, abs=0.01)
        assert env.jetson_compressed_mb == pytest.approx(0.5, abs=0.01)

    def test_payload_send_transfers_data_to_obc(self):
        """payload_send mode transfers compressed data from Jetson to OBC via RS-485."""
        env = self._make_pipeline_env(compression_ratio=2.0)
        env.step({"eventsat_0": {"mode": "payload_observe"}})
        # After 2 compress steps, data is in jetson_compressed
        env.step({"eventsat_0": {"mode": "payload_compress"}})
        env.step({"eventsat_0": {"mode": "payload_compress"}})
        compressed_before = env.jetson_compressed_mb
        assert compressed_before > 0
        # payload_send moves data to OBC at 50 kbps rate
        result = env.step({"eventsat_0": {"mode": "payload_send"}})
        assert env.obc_data_mb > 0.0
        assert env.jetson_compressed_mb < compressed_before
        assert result.info.get("had_data_to_send") is True
        assert result.info.get("data_sent_mb", 0.0) > 0.0

    def test_communication_only_downlinks_from_obc(self):
        """Communication only downlinks data that is on OBC."""
        env = self._make_pipeline_env()
        # Force some data into jetson_raw (but not OBC yet)
        env.jetson_raw_mb = 10.0
        env.data_stored_mb = 10.0
        env.uncompressed_observations = 5
        # Force a ground pass
        env._orbital_ctx = None
        env.current_step = 0  # simplified pass check fallback
        # Try to downlink — should get 0 since OBC is empty
        initial_dl = env.data_downlinked_mb
        # No pass active in default fallback, so test with manual obc manipulation
        env.obc_data_mb = 5.0
        env.data_stored_mb = 15.0
        # With no ground pass, comm falls back to charging
        result = env.step({"eventsat_0": {"mode": "communication"}})
        # Resolved to charging (no pass), so no data downlinked
        assert env.data_downlinked_mb == pytest.approx(initial_dl, abs=0.01)

    def test_data_stored_mb_is_total(self):
        """data_stored_mb = jetson_raw + jetson_compressed + obc_data."""
        env = self._make_pipeline_env()
        env.jetson_raw_mb = 4.0
        env.jetson_compressed_mb = 1.0
        env.obc_data_mb = 2.0
        # Run one step to trigger data_stored_mb update
        env.step({"eventsat_0": {"mode": "charging"}})
        assert env.data_stored_mb == pytest.approx(
            env.jetson_raw_mb + env.jetson_compressed_mb + env.obc_data_mb, abs=0.01
        )

    def test_compression_ratio_6_7(self):
        """Test with PDR compression ratio: 85% reduction = 6.7:1."""
        env = self._make_pipeline_env(compression_ratio=6.7)
        env.step({"eventsat_0": {"mode": "payload_observe"}})
        env.step({"eventsat_0": {"mode": "payload_compress"}})
        env.step({"eventsat_0": {"mode": "payload_compress"}})
        # 2.0 MB / 6.7 ≈ 0.299 MB
        assert env.jetson_compressed_mb == pytest.approx(2.0 / 6.7, abs=0.01)


# -----------------------------------------------------------------
# Thermal model removal
# -----------------------------------------------------------------

class TestThermalRemoved:
    """Verify env works correctly after thermal model removal."""

    def test_no_thermal_state(self):
        """Environment should not have thermal state attributes."""
        env = _make_env_with_scenario()
        assert not hasattr(env, "jetson_temp_c")
        assert not hasattr(env, "thermal_cooldown")
        assert not hasattr(env, "max_temp_c")

    def test_continuous_observation_not_blocked(self):
        """Without thermal model, observation is not blocked by temperature."""
        env = _make_env_with_scenario()
        # Force high SoC so battery doesn't constrain
        env.battery_soc = 0.95
        # Should be able to observe many steps without thermal shutdown
        obs_count = 0
        for _ in range(30):
            result = env.step({"eventsat_0": {"mode": "payload_observe"}})
            if result.info["resolved_mode"] == "payload_observe":
                obs_count += 1
        # All 30 steps should be observation (no thermal block)
        assert obs_count == 30

    def test_no_thermal_in_observation_metadata(self):
        """Observation metadata should not include thermal fields."""
        env = _make_env_with_scenario()
        obs = env.get_observation()
        sat = obs.constellation_state.satellites["eventsat_0"]
        assert "jetson_temp_c" not in sat.metadata
        assert "thermal_cooldown" not in sat.metadata

    def test_no_thermal_in_step_info(self):
        """Step info should not include thermal fields."""
        env = _make_env_with_scenario()
        result = env.step({"eventsat_0": {"mode": "payload_observe"}})
        assert "jetson_temp_c" not in result.info
        assert "thermal_cooldown" not in result.info


# -----------------------------------------------------------------
# Backward compatibility
# -----------------------------------------------------------------

class TestBackwardCompat:
    """Default config (no new params) should not crash and behave sensibly."""

    def test_env_works_without_new_params(self):
        """Env with minimal config (no compression_ratio, no transition_overhead) works."""
        env = _make_env_with_scenario()
        assert env.compression_ratio == 1.0
        assert env.jetson_to_obc_rate_kbps == 50  # RS-485 default
        assert env.settling_time_steps == 0
        env.reset(seed=1)
        for _ in range(10):
            env.step({"eventsat_0": {"mode": "payload_observe"}})
        assert env.total_observation_s > 0

    def test_no_automatic_transfer(self):
        """Compressed data stays on Jetson until payload_send is used."""
        env = _make_env_with_scenario()
        env.step({"eventsat_0": {"mode": "payload_observe"}})
        env.step({"eventsat_0": {"mode": "payload_compress"}})
        env.step({"eventsat_0": {"mode": "payload_compress"}})
        # Charging does NOT transfer data — requires explicit payload_send
        env.step({"eventsat_0": {"mode": "charging"}})
        assert env.jetson_compressed_mb > 0  # Data still on Jetson
        assert env.obc_data_mb == pytest.approx(0.0, abs=0.01)

    def test_full_episode_runs(self):
        """A complete episode runs without errors."""
        env = _make_env_with_scenario()
        env.reset(seed=42)
        steps = 0
        while not env.is_done():
            env.step({"eventsat_0": {"mode": "payload_observe"}})
            steps += 1
        assert steps == 500


# -----------------------------------------------------------------
# Detection mode (C2)
# -----------------------------------------------------------------

class TestDetectionMode:
    """C2: payload_detect mode — multi-step CV detection on compressed data."""

    def test_detection_takes_multiple_steps(self):
        """With detection_time_s=300 and step=60s, detection takes 5 steps."""
        env = _make_env_with_scenario(storage={"compression_ratio": 1.0})
        env.battery_soc = 0.95
        # Observe → compress × 2 → now have 1 undetected observation
        env.step({"eventsat_0": {"mode": "payload_observe"}})
        env.step({"eventsat_0": {"mode": "payload_compress"}})
        env.step({"eventsat_0": {"mode": "payload_compress"}})
        assert env.undetected_observations == 1

        # Detection should take 5 steps (300s / 60s)
        for i in range(4):
            result = env.step({"eventsat_0": {"mode": "payload_detect"}})
            assert env.undetected_observations == 1  # not done yet
            assert env.detection_progress == i + 1

        # 5th step completes detection
        result = env.step({"eventsat_0": {"mode": "payload_detect"}})
        assert env.undetected_observations == 0
        assert env.total_detections == 1
        assert env.detection_progress == 0
        assert result.info.get("detection_completed") is True

    def test_detection_progress_resets_on_mode_switch(self):
        """Switching away from detect resets detection progress."""
        env = _make_env_with_scenario(storage={"compression_ratio": 1.0})
        env.battery_soc = 0.95
        env.step({"eventsat_0": {"mode": "payload_observe"}})
        env.step({"eventsat_0": {"mode": "payload_compress"}})
        env.step({"eventsat_0": {"mode": "payload_compress"}})
        # Start detecting
        env.step({"eventsat_0": {"mode": "payload_detect"}})
        assert env.detection_progress == 1
        # Switch away
        env.step({"eventsat_0": {"mode": "charging"}})
        assert env.detection_progress == 0

    def test_no_detection_without_data(self):
        """Detection with no undetected observations does nothing useful."""
        env = _make_env_with_scenario()
        env.battery_soc = 0.95
        result = env.step({"eventsat_0": {"mode": "payload_detect"}})
        assert result.info.get("had_data_to_detect") is False
        assert env.total_detections == 0

    def test_detection_produces_metadata_on_obc(self):
        """Completed detection adds small metadata to OBC."""
        env = _make_env_with_scenario(storage={"compression_ratio": 1.0})
        env.battery_soc = 0.95
        # Full pipeline: observe → compress × 2 → detect × 5
        env.step({"eventsat_0": {"mode": "payload_observe"}})
        env.step({"eventsat_0": {"mode": "payload_compress"}})
        env.step({"eventsat_0": {"mode": "payload_compress"}})
        obc_before = env.obc_data_mb
        for _ in range(5):
            env.step({"eventsat_0": {"mode": "payload_detect"}})
        # OBC should have increased by detection_metadata_mb (0.01)
        assert env.obc_data_mb > obc_before

    def test_detect_mode_in_valid_modes(self):
        """payload_detect is accepted as a valid mode."""
        env = _make_env_with_scenario()
        env.battery_soc = 0.95
        result = env.step({"eventsat_0": {"mode": "payload_detect"}})
        assert result.info["resolved_mode"] == "payload_detect"

    def test_detect_blocked_by_low_battery(self):
        """Detection is blocked when SoC < detect_min_soc."""
        env = _make_env_with_scenario()
        env.battery_soc = 0.25  # below 0.3 threshold
        result = env.step({"eventsat_0": {"mode": "payload_detect"}})
        assert result.info["resolved_mode"] == "charging"


# -----------------------------------------------------------------
# Pipeline backpressure (C3)
# -----------------------------------------------------------------

class TestPipelineBackpressure:
    """C3: Agent pipeline saturation rule and configurable downlink budget."""

    def test_daily_downlink_budget_in_observation(self):
        """daily_downlink_budget_mb is exposed in observation metadata."""
        env = _make_env_with_scenario()
        obs = env.get_observation()
        sat = obs.constellation_state.satellites["eventsat_0"]
        assert sat.metadata["daily_downlink_budget_mb"] == 27.0

    def test_undetected_observations_in_observation(self):
        """undetected_observations is exposed in observation metadata."""
        env = _make_env_with_scenario()
        obs = env.get_observation()
        sat = obs.constellation_state.satellites["eventsat_0"]
        assert "undetected_observations" in sat.metadata

    def test_agent_rule_pipeline_saturation(self):
        """R5b: Agent charges when pipeline exceeds daily downlink budget."""
        from src.representation.rule_based_eventsat import RuleBasedEventSat
        agent = RuleBasedEventSat()
        # Pipeline data exceeds budget
        state = {
            "battery_soc": 0.9,
            "ground_pass_active": False,
            "data_stored_mb": 50.0,
            "obc_data_mb": 20.0,
            "jetson_compressed_mb": 10.0,  # Total pipeline = 30 > 27
            "storage_capacity_mb": 512.0,
            "uncompressed_observations": 0,
            "undetected_observations": 0,
            "health_status": "nominal",
            "daily_downlink_budget_mb": 27.0,
        }
        action = agent.select_action(DecisionContext(state=state))
        assert action["eventsat_0"]["mode"] == "charging"
        assert "R5b" in agent.get_rationale()

    def test_agent_rule_detect_when_undetected(self):
        """R5c: Agent detects when there are undetected observations."""
        from src.representation.rule_based_eventsat import RuleBasedEventSat
        agent = RuleBasedEventSat()
        state = {
            "battery_soc": 0.9,
            "ground_pass_active": False,
            "data_stored_mb": 5.0,
            "obc_data_mb": 2.0,
            "jetson_compressed_mb": 1.0,
            "storage_capacity_mb": 512.0,
            "uncompressed_observations": 0,
            "undetected_observations": 3,
            "health_status": "nominal",
            "daily_downlink_budget_mb": 27.0,
        }
        action = agent.select_action(DecisionContext(state=state))
        assert action["eventsat_0"]["mode"] == "payload_detect"
        assert "R5c" in agent.get_rationale()


# -----------------------------------------------------------------
# Pipeline efficiency metric (C4)
# -----------------------------------------------------------------

class TestPipelineEfficiency:
    """C4: Dynamic max utility metric — data downlink efficiency."""

    def test_pass_duration_tracking(self):
        """total_pass_duration_s increments during ground passes."""
        env = _make_env_with_scenario()
        env.battery_soc = 0.95
        # No pass → no increase
        env.step({"eventsat_0": {"mode": "charging"}})
        assert env.total_pass_duration_s == 0.0

    def test_max_achievable_downlink_in_step_info(self):
        """max_achievable_downlink_mb appears in step info."""
        env = _make_env_with_scenario()
        result = env.step({"eventsat_0": {"mode": "charging"}})
        assert "max_achievable_downlink_mb" in result.info

    def test_data_downlink_efficiency_computation(self):
        """Data downlink efficiency = downlinked / max_achievable."""
        from src.orchestration.eventsat_metrics import EventSatMetricsCollector
        from src.orchestration.metrics_collector import StepMetrics

        collector = EventSatMetricsCollector(config={
            "utility_targets": {
                "observation_hours": 2.0,
                "downlinked_mb": 240.0,
                "mission_duration_days": 90.0,
            },
        })
        # Create fake step metrics with known values
        steps = []
        for i in range(10):
            steps.append(StepMetrics(
                timestep=i,
                wall_clock_seconds=0.01,
                reward=0.1,
                metrics={
                    "battery_soc": 0.8,
                    "data_stored_mb": 5.0,
                    "data_downlinked_mb": 50.0 if i == 9 else 0.0,
                    "observation_hours": 1.0 if i == 9 else 0.0,
                    "in_sunlight": 1.0,
                    "ground_pass_active": 0.0,
                    "forced": 0.0,
                    "anomaly": 0.0,
                    "safety_override": 0.0,
                    "jetson_raw_mb": 0.0,
                    "jetson_compressed_mb": 0.0,
                    "obc_data_mb": 0.0,
                    "in_transition": False,
                    "energy_consumed_wh": 0.1,
                    "decision_latency_s": 0.001,
                    "has_rationale": 1.0,
                    "total_detections": 5,
                    "max_achievable_downlink_mb": 100.0 if i == 9 else 0.0,
                },
            ))
        episode = collector.aggregate_episode_metrics(steps)
        # data_downlink_efficiency = 50 / 100 = 0.5
        assert episode.aggregated["data_downlink_efficiency"] == pytest.approx(0.5)
        assert episode.aggregated["total_detections"] == 5.0


# -----------------------------------------------------------------
# Payload send mode (RS-485 Jetson→OBC)
# -----------------------------------------------------------------

class TestPayloadSend:
    """payload_send: explicit RS-485 transfer from Jetson to OBC."""

    def test_send_transfers_at_rate(self):
        """50 kbps = 0.375 MB per 60s step."""
        env = _make_env_with_scenario(storage={"compression_ratio": 1.0})
        env.battery_soc = 0.95
        # Full pipeline: observe → compress × 2
        env.step({"eventsat_0": {"mode": "payload_observe"}})
        env.step({"eventsat_0": {"mode": "payload_compress"}})
        env.step({"eventsat_0": {"mode": "payload_compress"}})
        compressed = env.jetson_compressed_mb  # 2.0 MB (ratio=1.0)
        # One send step: 50 kbps × 60s = 0.375 MB
        result = env.step({"eventsat_0": {"mode": "payload_send"}})
        expected_transfer = (50 / 8.0) * (60 / 1000.0)  # 0.375 MB
        assert result.info["data_sent_mb"] == pytest.approx(expected_transfer, abs=0.01)
        assert env.jetson_compressed_mb == pytest.approx(compressed - expected_transfer, abs=0.01)
        assert env.obc_data_mb == pytest.approx(expected_transfer, abs=0.02)

    def test_send_multiple_steps_drains_jetson(self):
        """Enough send steps transfer all compressed data to OBC."""
        env = _make_env_with_scenario(storage={"compression_ratio": 2.0})
        env.battery_soc = 0.95
        env.step({"eventsat_0": {"mode": "payload_observe"}})
        env.step({"eventsat_0": {"mode": "payload_compress"}})
        env.step({"eventsat_0": {"mode": "payload_compress"}})
        # 2.0 / 2.0 = 1.0 MB compressed. At 0.375 MB/step → ceil(1.0/0.375) = 3 steps
        for _ in range(3):
            env.step({"eventsat_0": {"mode": "payload_send"}})
        assert env.jetson_compressed_mb == pytest.approx(0.0, abs=0.01)
        assert env.obc_data_mb == pytest.approx(1.0, abs=0.02)

    def test_send_no_data_does_nothing(self):
        """Sending with no compressed data on Jetson is a failed action."""
        env = _make_env_with_scenario()
        env.battery_soc = 0.95
        result = env.step({"eventsat_0": {"mode": "payload_send"}})
        assert result.info["had_data_to_send"] is False
        assert result.info["data_sent_mb"] == 0.0

    def test_send_blocked_by_low_battery(self):
        """payload_send is blocked when SoC < send_min_soc."""
        env = _make_env_with_scenario()
        env.battery_soc = 0.25
        result = env.step({"eventsat_0": {"mode": "payload_send"}})
        assert result.info["resolved_mode"] == "charging"

    def test_send_is_valid_mode(self):
        """payload_send is accepted as a valid mode."""
        env = _make_env_with_scenario()
        env.battery_soc = 0.95
        result = env.step({"eventsat_0": {"mode": "payload_send"}})
        assert result.info["resolved_mode"] == "payload_send"

    def test_agent_rule_send_when_compressed(self):
        """R5d: Agent sends when Jetson has compressed data."""
        from src.representation.rule_based_eventsat import RuleBasedEventSat
        agent = RuleBasedEventSat()
        state = {
            "battery_soc": 0.9,
            "ground_pass_active": False,
            "data_stored_mb": 5.0,
            "obc_data_mb": 0.0,
            "jetson_compressed_mb": 1.5,
            "storage_capacity_mb": 512.0,
            "uncompressed_observations": 0,
            "undetected_observations": 0,
            "health_status": "nominal",
            "daily_downlink_budget_mb": 27.0,
        }
        action = agent.select_action(DecisionContext(state=state))
        assert action["eventsat_0"]["mode"] == "payload_send"
        assert "R5d" in agent.get_rationale()


# -----------------------------------------------------------------
# Anomaly: forced safe mode + ground recovery
# -----------------------------------------------------------------


class TestAnomalyForcedSafe:
    """Verify that active anomalies are environment-enforced (not agent-voluntary)."""

    def test_forced_safe_on_anomaly(self):
        """_resolve_mode must return 'safe' for any request when anomaly is active."""
        env = _make_env()
        env.active_anomaly = "thermal_warning"
        env.forced_safe_steps = 5

        for mode in ["charging", "payload_observe", "communication", "payload_compress"]:
            assert env._resolve_mode(mode) == "safe", (
                f"Expected 'safe' when anomaly active, got something else for mode={mode}"
            )

    def test_no_forced_safe_without_anomaly(self):
        """_resolve_mode must NOT force safe when no anomaly is active."""
        env = _make_env()
        env.active_anomaly = None
        # charging is always valid
        assert env._resolve_mode("charging") == "charging"

    def test_anomaly_step_forces_safe_mode(self):
        """Environment step must execute safe mode even when agent requests charging."""
        env = _make_env(anomaly_prob=0.0)
        # Manually inject anomaly
        env.active_anomaly = "thermal_warning"
        env.forced_safe_steps = 5

        result = env.step({"eventsat_0": {"mode": "charging"}})
        assert result.info["resolved_mode"] == "safe", (
            "Environment should force safe mode during active anomaly"
        )

    def test_anomaly_recovery_requires_ground_pass(self):
        """Anomaly must NOT clear when countdown expires but no ground pass is active."""
        env = _make_env(anomaly_prob=0.0)
        env.active_anomaly = "thermal_warning"
        env.forced_safe_steps = 1  # Will hit 0 after one decrement

        # Ensure no ground pass is active (simplified model, no passes in minimal env)
        # _orbital_ctx is set from reset(seed=0), which uses simplified model with random passes.
        # Override to guarantee no pass at current step.
        env._orbital_ctx = None  # Falls back to: return False in _is_ground_pass_active

        # Step: countdown goes to 0 but no ground pass → anomaly persists
        env._maybe_inject_anomaly()
        assert env.active_anomaly is not None, (
            "Anomaly should persist when countdown expired but no ground pass is active"
        )

    def test_anomaly_clears_on_ground_pass(self):
        """Anomaly must clear when countdown expires AND ground pass is active."""
        from unittest.mock import patch
        env = _make_env(anomaly_prob=0.0)
        env.active_anomaly = "thermal_warning"
        env.forced_safe_steps = 1

        # Mock ground pass to be active
        with patch.object(env, "_is_ground_pass_active", return_value=True):
            env._maybe_inject_anomaly()

        assert env.active_anomaly is None, (
            "Anomaly should clear when countdown expired and ground pass is active"
        )

    def test_anomaly_forced_safe_in_step_metrics(self):
        """anomaly_forced_safe must be 1.0 in step metrics while anomaly is active."""
        env = _make_env(anomaly_prob=0.0)
        env.active_anomaly = "thermal_warning"
        env.forced_safe_steps = 5

        result = env.step({"eventsat_0": {"mode": "charging"}})
        assert result.info.get("anomaly_forced_safe") == 1.0

    def test_no_anomaly_forced_safe_metric_when_nominal(self):
        """anomaly_forced_safe must be 0.0 when no anomaly is active."""
        env = _make_env(anomaly_prob=0.0)
        assert env.active_anomaly is None

        result = env.step({"eventsat_0": {"mode": "charging"}})
        assert result.info.get("anomaly_forced_safe") == 0.0
