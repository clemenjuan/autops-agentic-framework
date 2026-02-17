"""
CoALA Reasoning Engine

Implements the CoALA (Cognitive Architectures for Language Agents) framework
with pure Planning ↔ Execution cycles.

Reference: https://arxiv.org/abs/2309.02427

Architecture:
- Planning: Internal actions (reasoning + retrieval) to select next action
- Execution: External actions (tools) or learning actions (memory writes)
- Iterates until task complete

Replaces old Think→Plan→Execute→Reflect with cleaner CoALA structure.
"""

import json
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum

from agent.coala_action_space import CoALAActionSpace, ActionType
from agent.memory import WorkingMemory, EpisodicMemory, SemanticMemory, ProceduralMemory
from utils.toon_formatter import ToonFormatter


class CoALAState(Enum):
    """States in CoALA decision-making cycle."""
    INITIAL = "initial"
    PLANNING = "planning"
    EXECUTION = "execution"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class CycleStep:
    """Represents one step in the CoALA reasoning cycle."""
    cycle_number: int
    state: CoALAState
    timestamp: str
    action_selected: Optional[str]
    action_type: Optional[str]
    reasoning: str
    results: Dict[str, Any]
    confidence: float
    
    def to_dict(self):
        return {
            'cycle': self.cycle_number,
            'state': self.state.value,
            'timestamp': self.timestamp,
            'action_selected': self.action_selected,
            'action_type': self.action_type,
            'reasoning': self.reasoning,
            'results': self.results,
            'confidence': self.confidence
        }


class CoALAReasoningEngine:
    """
    CoALA Reasoning Engine implementing Planning ↔ Execution cycles.
    
    Uses four memory modules (working, episodic, semantic, procedural)
    and structured action space (internal vs external actions).
    """
    
    def __init__(
        self,
        reasoning_llm,
        general_llm,
        tools: Dict,
        tools_metadata: Dict,
        working_memory: WorkingMemory,
        episodic_memory: EpisodicMemory,
        semantic_memory: SemanticMemory,
        procedural_memory: ProceduralMemory,
        max_cycles: int = 15
    ):
        """
        Initialize CoALA reasoning engine.
        
        Args:
            reasoning_llm: LLM for complex reasoning
            general_llm: LLM for quick tasks
            tools: Dictionary of available tools
            tools_metadata: Tool metadata
            working_memory: Working memory module
            episodic_memory: Episodic memory module
            semantic_memory: Semantic memory module
            procedural_memory: Procedural memory module
            max_cycles: Maximum number of planning-execution cycles
        """
        self.reasoning_llm = reasoning_llm
        self.general_llm = general_llm
        self.tools = tools
        self.tools_metadata = tools_metadata
        self.max_cycles = max_cycles
        
        self.working_memory = working_memory
        self.episodic_memory = episodic_memory
        self.semantic_memory = semantic_memory
        self.procedural_memory = procedural_memory
        
        memory_modules = {
            'working': self.working_memory,
            'episodic': self.episodic_memory,
            'semantic': self.semantic_memory,
            'procedural': self.procedural_memory
        }
        
        self.action_space = CoALAActionSpace(tools=tools, memory_modules=memory_modules)
        
        self.cycle_history: List[CycleStep] = []
        self.current_cycle = 0
        self.original_task = ""
        self.task_keywords = []
    
    async def reason(self, situation_data: Dict) -> Dict:
        """
        Main reasoning loop using CoALA cycles.
        
        Args:
            situation_data: Dictionary with 'task_description' and optional context
            
        Returns:
            Dictionary with final results and reasoning trace
        """
        self.original_task = situation_data.get('task_description', '')
        self.cycle_history = []
        self.current_cycle = 0
        
        self.working_memory.reset()
        self.working_memory.set_current_task(self.original_task)
        
        # Preprocess task with general LLM to extract keywords and categorize
        await self._preprocess_task()
        
        print(f"\n[CoALA Engine] Starting task: {self.original_task}")
        print(f"[CoALA Engine] Extracted keywords: {self.task_keywords}")
        print(f"[CoALA Engine] Action space: {len(self.action_space.get_internal_actions())} internal, {len(self.action_space.get_external_actions())} external")
        
        current_state = CoALAState.INITIAL
        
        while self.current_cycle < self.max_cycles:
            try:
                if current_state == CoALAState.INITIAL:
                    current_state = CoALAState.PLANNING
                    self.current_cycle = 1
                    print(f"\n[CoALA Engine] ===== Cycle {self.current_cycle} =====")
                
                elif current_state == CoALAState.PLANNING:
                    action_to_execute = await self._planning_cycle()
                    
                    if action_to_execute is None:
                        print(f"[CoALA Engine] Planning decided task is complete")
                        current_state = CoALAState.COMPLETED
                    else:
                        current_state = CoALAState.EXECUTION
                
                elif current_state == CoALAState.EXECUTION:
                    should_continue = await self._execution_cycle()
                    
                    if should_continue and self.current_cycle < self.max_cycles:
                        self.current_cycle += 1
                        print(f"\n[CoALA Engine] ===== Cycle {self.current_cycle} =====")
                        current_state = CoALAState.PLANNING
                    else:
                        current_state = CoALAState.COMPLETED
                
                elif current_state == CoALAState.COMPLETED:
                    break
                
                else:
                    break
            
            except Exception as e:
                print(f"[CoALA Engine] ERROR in cycle {self.current_cycle}: {e}")
                import traceback
                traceback.print_exc()
                current_state = CoALAState.ERROR
                break
        
        final_result = await self._synthesize_final_result()
        
        await self._store_episode(final_result)
        
        print(f"[CoALA Engine] Completed after {self.current_cycle} cycle(s)")
        
        return final_result
    
    async def _planning_cycle(self) -> Optional[str]:
        """
        Planning cycle: Use internal actions to select next action.
        
        Steps:
        1. Retrieve from long-term memories (episodic, semantic, procedural)
        2. Reason with LLM using working memory + retrieved context
        3. Propose candidate actions
        4. Evaluate and select best action
        
        Returns:
            Name of action to execute, or None if task complete
        """
        print(f"[CoALA Planning] Retrieving from long-term memories...")
        
        # Use preprocessed keywords from general LLM (fallback to simple split if not available)
        task_keywords = self.task_keywords if self.task_keywords else self.original_task.lower().split()
        
        past_episodes = self.episodic_memory.retrieve({
            'task_keywords': task_keywords
        }, limit=3)
        
        relevant_facts = self.semantic_memory.retrieve({
            'keywords': task_keywords
        }, limit=5)
        
        relevant_procedures = self.procedural_memory.retrieve({
            'context_keywords': task_keywords
        }, limit=3)
        
        self.working_memory.add_intermediate_result('retrieved_episodes', len(past_episodes))
        self.working_memory.add_intermediate_result('retrieved_facts', len(relevant_facts))
        self.working_memory.add_intermediate_result('retrieved_procedures', len(relevant_procedures))
        
        print(f"[CoALA Planning] Retrieved: {len(past_episodes)} episodes, {len(relevant_facts)} facts, {len(relevant_procedures)} procedures")
        
        memory_context = self._format_memory_context(past_episodes, relevant_facts, relevant_procedures)
        
        available_tools = [tool for tool in self.tools.keys()]
        tools_description = self._format_tools_for_llm()
        
        current_state = self.working_memory.get_current_state()
        
        planning_prompt = f"""
        You are a satellite operations agent using CoALA (Cognitive Architectures for Language Agents).
        
        CURRENT TASK: {self.original_task}
        
        CYCLE: {self.current_cycle}/{self.max_cycles}
        
        WORKING MEMORY STATE:
        {ToonFormatter.dumps(current_state)}
        
        RETRIEVED CONTEXT FROM LONG-TERM MEMORY:
        {memory_context}
        
        AVAILABLE TOOLS (External Actions):
        {tools_description}
        
        Your task is to select the NEXT ACTION to take. Consider:
        1. What have we done so far? (check working memory)
        2. What did similar past tasks do? (episodic memory)
        3. What facts might help? (semantic memory)
        4. What strategies have worked before? (procedural memory)
        5. Should we execute a tool, or is the task complete?
        
        NOTE: Context data is provided in TOON format (Token-Oriented Object Notation, more compact than JSON).
        
        CRITICAL: If required tools are not implemented or giving weird results, set task_complete=true and explain why blocked.
        
        Respond with TOON format (or JSON if TOON is not available):
        {{
            "analysis": "Brief analysis of current situation",
            "next_action": "tool_name OR null if task complete/blocked",
            "parameters": {{"param": "value"}},
            "reasoning": "Why this action? Or why blocked?",
            "confidence": 0.8,
            "task_complete": false
        }}
        
        IMPORTANT: confidence must be 0.0-1.0. Set task_complete=true if no more actions OR if blocked.
        """
        
        response = await self.reasoning_llm.reason(planning_prompt, show_thinking=True)
        print(f"[CoALA Planning] LLM response received")
        
        plan = self._parse_data(response)
        
        if 'error' in plan:
            print(f"[CoALA Planning] Data parse failed, retrying...")
            plan = await self._parse_data_with_retry(response, self.reasoning_llm, planning_prompt, total_attempts=1)
        
        next_action = plan.get('next_action')
        task_complete = plan.get('task_complete', False)
        confidence = self._parse_confidence(plan.get('confidence', 0.5))
        
        self.working_memory.update_confidence(confidence)
        
        step = CycleStep(
            cycle_number=self.current_cycle,
            state=CoALAState.PLANNING,
            timestamp=datetime.now(timezone.utc).isoformat(),
            action_selected=next_action,
            action_type='grounding' if next_action else 'none',
            reasoning=plan.get('reasoning', ''),
            results={'plan': plan},
            confidence=confidence
        )
        self.cycle_history.append(step)
        
        if task_complete or next_action is None:
            print(f"[CoALA Planning] Task marked as complete")
            return None
        
        self.working_memory.add_intermediate_result('selected_action', next_action)
        self.working_memory.add_intermediate_result('action_parameters', plan.get('parameters', {}))
        
        print(f"[CoALA Planning] Selected action: {next_action}")
        
        return next_action
    
    async def _execution_cycle(self) -> bool:
        """
        Execution cycle: Execute the selected action.
        
        Can execute:
        - External actions (grounding tools)
        - Learning actions (store to memory)
        
        Returns:
            True if should continue to next cycle, False if done
        """
        current_state = self.working_memory.get_current_state()
        action_name = current_state.get('intermediate_results', {}).get('selected_action')
        action_params = current_state.get('intermediate_results', {}).get('action_parameters', {})
        
        if not action_name:
            print(f"[CoALA Execution] No action to execute")
            return False
        
        print(f"[CoALA Execution] Executing: {action_name}")
        
        action = self.action_space.get_action(action_name)
        
        if action is None:
            print(f"[CoALA Execution] ERROR: Action '{action_name}' not found")
            return False
        
        try:
            if action.is_external():
                tool_func = self.tools[action_name]['execute']
                result = await tool_func(action_params)
                print(f"[CoALA Execution] Tool result: {result}")
                
                self.working_memory.add_intermediate_result(f'tool_result_{action_name}', result)
                
                if action_name == 'region_mapper' and 'bbox' in result:
                    self.semantic_memory.store_region_info(
                        region_name=action_params.get('region_name', 'unknown'),
                        bbox=result.get('bbox', []),
                        center=result.get('center', []),
                        metadata=result
                    )
                    print(f"[CoALA Execution] Stored region info to semantic memory")
            
            elif action.is_internal() and action.action_type == ActionType.INTERNAL_LEARNING:
                result = self.action_space.execute_action(action_name, action_params)
                print(f"[CoALA Execution] Stored to memory: {action_name}")
            
            else:
                result = {'status': 'unsupported_action_type'}
            
            step = CycleStep(
                cycle_number=self.current_cycle,
                state=CoALAState.EXECUTION,
                timestamp=datetime.now(timezone.utc).isoformat(),
                action_selected=action_name,
                action_type=action.action_type.value,
                reasoning=f"Executed {action_name}",
                results={'action_result': result},
                confidence=0.8
            )
            self.cycle_history.append(step)
            
            return self.current_cycle < self.max_cycles
        
        except Exception as e:
            print(f"[CoALA Execution] ERROR executing {action_name}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def _store_episode(self, final_result: Dict):
        """Store the complete episode to episodic memory."""
        actions_taken = []
        for step in self.cycle_history:
            if step.action_selected:
                actions_taken.append({
                    'action': step.action_selected,
                    'type': step.action_type,
                    'cycle': step.cycle_number
                })
        
        episode = {
            'task': self.original_task,
            'actions': actions_taken,
            'results': final_result.get('tool_results', {}),
            'outcome': final_result.get('task_status', 'completed'),
            'confidence': final_result.get('confidence', 0.5),
            'reasoning_trace': [step.to_dict() for step in self.cycle_history],
            'cycles': self.current_cycle
        }
        
        episode_id = self.episodic_memory.store(episode)
        print(f"[CoALA Engine] Stored episode to episodic memory: {episode_id}")
    
    async def _synthesize_final_result(self) -> Dict:
        """Synthesize final result from cycle history."""
        tool_results = {}
        actions_executed = []
        
        for step in self.cycle_history:
            if step.state == CoALAState.EXECUTION and step.action_selected:
                actions_executed.append(step.action_selected)
                if 'action_result' in step.results:
                    tool_results[step.action_selected] = step.results['action_result']
        
        final_confidence = self.working_memory.get_current_state().get('confidence', 0.5)
        
        synthesis_prompt = f"""
        Synthesize the final result of this satellite operations task.
        
        ORIGINAL TASK: {self.original_task}
        
        ACTIONS EXECUTED: {actions_executed}
        
        TOOL RESULTS: {ToonFormatter.dumps(tool_results)}
        
        CYCLES COMPLETED: {self.current_cycle}/{self.max_cycles}
        
        Provide a comprehensive summary in TOON format (or JSON if TOON is not available):
        {{
            "situation_summary": "Clear summary of what was accomplished",
            "analysis": "Key findings and insights",
            "recommendations": ["actionable", "recommendations"],
            "confidence": {final_confidence},
            "task_status": "completed"
        }}
        """
        
        response = await self.reasoning_llm.reason(synthesis_prompt, show_thinking=False)
        final_result = self._parse_data(response)
        
        if 'error' in final_result:
            print(f"[CoALA Synthesis] Data parse failed, retrying...")
            final_result = await self._parse_data_with_retry(response, self.reasoning_llm, synthesis_prompt, total_attempts=1)
        
        if 'error' in final_result:
            final_result = {
                'situation_summary': f"Completed task: {self.original_task}",
                'analysis': f"Executed {len(actions_executed)} actions",
                'recommendations': ["Review results"],
                'confidence': final_confidence,
                'task_status': 'completed'
            }
        
        final_result['reasoning_trace'] = [step.to_dict() for step in self.cycle_history]
        final_result['tool_results'] = tool_results
        final_result['total_cycles'] = self.current_cycle
        final_result['actions_executed'] = actions_executed
        
        return final_result
    
    async def _preprocess_task(self):
        """
        Preprocess task using general LLM to extract keywords and categorize.
        This improves memory retrieval accuracy by using semantic understanding
        instead of simple word splitting.
        """
        if not self.original_task:
            self.task_keywords = []
            return
        
        try:
            preprocessing_prompt = f"""
            Analyze this satellite operations task and extract relevant keywords for memory retrieval.
            
            TASK: {self.original_task}
            
            Extract:
            1. Key entities (locations, objects, actions, time references)
            2. Domain-specific terms (satellite operations, Earth observation, etc.)
            3. Task type keywords
            
            Respond with TOON format (or JSON if TOON is not available):
            {{
                "keywords": ["keyword1", "keyword2", "keyword3"],
                "task_category": "brief category description",
                "entities": {{
                    "locations": ["location1"],
                    "objects": ["object1"],
                    "actions": ["action1"]
                }}
            }}
            
            IMPORTANT: Return ONLY valid TOON or JSON, no other text.
            """
            
            response = await self.general_llm.reason(preprocessing_prompt, show_thinking=False)
            preprocess_result = self._parse_data(response)
            
            if "error" not in preprocess_result and "keywords" in preprocess_result:
                self.task_keywords = preprocess_result.get("keywords", [])
                self.working_memory.add_intermediate_result('task_category', preprocess_result.get("task_category", ""))
                self.working_memory.add_intermediate_result('extracted_entities', preprocess_result.get("entities", {}))
                print(f"[CoALA Preprocessing] Extracted {len(self.task_keywords)} keywords using general LLM")
            else:
                # Fallback to simple split if LLM preprocessing fails
                self.task_keywords = self.original_task.lower().split()
                print(f"[CoALA Preprocessing] Fallback to simple keyword extraction")
        
        except Exception as e:
            print(f"[CoALA Preprocessing] Error during preprocessing: {e}. Using fallback.")
            self.task_keywords = self.original_task.lower().split()
    
    def _format_memory_context(self, episodes: List, facts: List, procedures: List) -> str:
        """Format retrieved memory context for LLM."""
        return ToonFormatter.dumps({
            "past_episodes": episodes[:3] if episodes else [],
            "relevant_facts": facts[:5] if facts else [],
            "known_strategies": procedures[:3] if procedures else []
        })
    
    def _format_tools_for_llm(self) -> str:
        """Format tool descriptions for LLM."""
        tools_list = []
        for tool in self.tools_metadata.get('tools', []):
            if not isinstance(tool, dict):
                continue
            tools_list.append({
                "name": tool.get('name', 'unknown'),
                "description": tool.get('description', ''),
                "parameters": tool.get('parameters', {})
            })
        return ToonFormatter.dumps(tools_list)
    
    def _parse_data(self, text: str) -> Dict:
        """Parse JSON or TOON from LLM response. Tries TOON first, then JSON."""
        import re
        
        text_clean = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        
        # Try TOON format first (looks for TOON block or TOON-like structure)
        toon_block_match = re.search(r'```(?:toon)?\s*([a-zA-Z_][a-zA-Z0-9_]*\[.*?\])\s*```', text_clean, re.DOTALL)
        if toon_block_match:
            try:
                toon_content = toon_block_match.group(1)
                return ToonFormatter.loads(toon_content)
            except Exception:
                pass
        
        # Try JSON block
        json_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text_clean, re.DOTALL)
        if json_block_match:
            try:
                return json.loads(json_block_match.group(1))
            except json.JSONDecodeError:
                # Try TOON parsing on the same content
                try:
                    return ToonFormatter.loads(json_block_match.group(1))
                except Exception:
                    pass
        
        # Try inline JSON/TOON
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text_clean, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                # Try TOON parsing
                try:
                    return ToonFormatter.loads(json_match.group(0))
                except Exception:
                    pass
        
        # Try TOON format (array-like structure)
        toon_match = re.search(r'([a-zA-Z_][a-zA-Z0-9_]*\[.*?\])', text_clean, re.DOTALL)
        if toon_match:
            try:
                return ToonFormatter.loads(toon_match.group(1))
            except Exception:
                pass
        
        return {"error": "Could not parse JSON or TOON", "raw_text": text}
    
    async def _parse_data_with_retry(self, text: str, llm, prompt: str, max_retries: int = 2, total_attempts: int = 1) -> Dict:
        """
        Parse JSON/TOON with retry logic. Uses general_llm after 3rd total attempt.
        
        Args:
            text: Original text to parse
            llm: Primary LLM to use (reasoning_llm)
            prompt: Original prompt (for context)
            max_retries: Maximum number of retry attempts
            total_attempts: Current total attempt count (including initial parse)
        """
        result = self._parse_data(text)
        if "error" not in result:
            return result
        
        for attempt in range(max_retries):
            current_attempt = total_attempts + attempt + 1
            
            # After 3rd total attempt, switch to general_llm for simpler retry
            retry_llm = self.general_llm if current_attempt >= 3 else llm
            
            if current_attempt >= 3:
                print(f"[CoALA Data Retry] Attempt {current_attempt}: Switching to general_llm")
            
            retry_prompt = """
            Your previous response could not be parsed. Provide ONLY valid TOON format (or JSON), no other text.
            Required format from original request. Ensure confidence is a number 0.0-1.0.
            """
            retry_response = await retry_llm.reason(retry_prompt, show_thinking=False)
            result = self._parse_data(retry_response)
            if "error" not in result:
                return result
        
        return result
    
    def _parse_confidence(self, confidence_raw) -> float:
        """Parse confidence value from various formats."""
        try:
            if isinstance(confidence_raw, str):
                import re
                numbers = re.findall(r'0\.\d+|\d+\.\d+', confidence_raw)
                if numbers:
                    return float(numbers[0])
                return 0.5
            return float(confidence_raw)
        except (ValueError, TypeError):
            return 0.5

