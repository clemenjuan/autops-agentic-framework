from flask import Flask, request, jsonify, render_template, send_file, Response, stream_with_context
from typing import Dict
import sys
import asyncio
from datetime import datetime, timezone
import os
import logging
import json
import threading
import time

from agent.llm_interface import LLMInterface
from agent.coala_reasoning_engine import CoALAReasoningEngine
from agent.memory import WorkingMemory, EpisodicMemory, SemanticMemory, ProceduralMemory
from tools.tool_loader import load_tools

class SatelliteOperationsAgent:
    def __init__(self):
        self.general_llm = LLMInterface(preferred_model="auto", role="general")
        self.reasoning_llm = LLMInterface(preferred_model="auto", role="reasoning")
        self.tools, self.tools_metadata = load_tools()
        
        self.working_memory = WorkingMemory(persistent=False)
        self.episodic_memory = EpisodicMemory()
        self.semantic_memory = SemanticMemory()
        self.procedural_memory = ProceduralMemory()
        
        self.reasoning_engine = CoALAReasoningEngine(
            reasoning_llm=self.reasoning_llm,
            general_llm=self.general_llm,
            tools=self.tools,
            tools_metadata=self.tools_metadata,
            working_memory=self.working_memory,
            episodic_memory=self.episodic_memory,
            semantic_memory=self.semantic_memory,
            procedural_memory=self.procedural_memory,
            max_cycles=15
        )
        self.mission_context = {}
        self.task_history = []
        
    async def process_query(self, query: str, additional_data: Dict = None) -> Dict:
        situation_data = {'task_description': query, 'mission_context': self.mission_context}
        if additional_data:
            situation_data.update(additional_data)
        
        result = await self.reasoning_engine.reason(situation_data)
        
        self.task_history.append({
            'id': len(self.task_history) + 1,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'query': query,
            'result': result,
            'reasoning_steps': result.get('reasoning_trace', []),
            'confidence': result.get('confidence', 0.0),
            'status': result.get('task_status', 'completed')
        })
        
        return result

app = Flask(__name__, static_folder='static', template_folder='templates')

agent = None
_initialized = False

class PrintLogger:
    def __init__(self, logger):
        self.logger = logger
    
    def write(self, message):
        if message.strip():
            self.logger.info(message.strip())
    
    def flush(self):
        pass

def initialize_app():
    global agent, _initialized
    if _initialized:
        return
    
    try:
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'autops_backend_{timestamp}.log')

        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))

        app.logger.addHandler(file_handler)
        app.logger.setLevel(logging.DEBUG)
        # sys.stdout = PrintLogger(app.logger) # Disable stdout hijacking to see errors in console

        app.logger.info('Initializing AUTOPS System...')
        agent = SatelliteOperationsAgent()
        app.logger.info('AUTOPS System initialized successfully')
        _initialized = True
        
    except Exception as e:
        app.logger.error(f"Failed to initialize app: {e}")
        # Ensure we don't mark as initialized if it failed
        _initialized = False
        raise e

@app.before_request
def ensure_initialized():
    initialize_app()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status', methods=['GET'])
def system_status():
    if agent is None:
        return jsonify({
            'status': 'initializing',
            'llm_status': 'Initializing...',
            'tools_available': [],
            'timestamp': datetime.utcnow().isoformat()
        })
    
    return jsonify({
        'status': 'online',
        'llm_status': agent.reasoning_llm.get_current_status(),
        'tools_available': list(agent.tools.keys()),
        'timestamp': datetime.utcnow().isoformat()
    })

@app.route('/api/query', methods=['POST'])
def handle_query():
    data = request.json or {}
    query = data.get('query') or data.get('task', '')
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    try:
        app.logger.info(f'=== NEW QUERY: {query} ================================================')
        
        additional_data = {k: v for k, v in data.items() if k not in ('query', 'task')}
        if additional_data:
            app.logger.info(f'Additional context: {additional_data}')
        
        result = asyncio.run(agent.process_query(query, additional_data))
        
        cycles = result.get("total_cycles", 0)
        app.logger.info(f'=== COMPLETED: Confidence {result.get("confidence", 0):.2f}, {cycles} cycle(s) ================================================')
        
        reasoning_log = []
        
        for step in result.get('reasoning_trace', []):
            if not isinstance(step, dict):
                continue
            
            state = step.get('state', 'unknown')
            cycle = step.get('cycle', 1)
            
            if state == 'planning':
                label = f'🧠 PLANNING (Cycle {cycle})'
            elif state == 'execution':
                label = f'⚙️ EXECUTION (Cycle {cycle})'
            else:
                label = state.upper()
            
            log_entry = {
                'step': label,
                'model': 'CoALA Engine',
                'summary': step.get('reasoning', ''),
                'confidence': step.get('confidence', 0.0),
                'action': step.get('action_selected', ''),
                'results': step.get('results', {})
            }
            
            reasoning_log.append(log_entry)
        
        return jsonify({
            'status': 'success',
            'query': query,
            'result': result.get('situation_summary', ''),
            'analysis': result.get('analysis', ''),
            'recommendations': result.get('recommendations', []),
            'confidence': result.get('confidence', 0.0),
            'reasoning_log': reasoning_log,
            'tool_results': result.get('tool_results', {}),
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except RuntimeError as e:
        error_msg = str(e)
        if 'LLM unavailable' in error_msg:
            app.logger.error(f'=== LLM UNAVAILABLE: {error_msg} ================================================')
            return jsonify({
                'status': 'error',
                'error': 'LLM Service Unavailable',
                'message': 'The language model service is currently unavailable.',
                'details': error_msg
            }), 503
        
        import traceback
        app.logger.error(f'=== QUERY FAILED ================================================\n{traceback.format_exc()}')
        return jsonify({
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500
    
    except Exception as e:
        import traceback
        app.logger.error(f'=== QUERY FAILED ================================================\n{traceback.format_exc()}')
        return jsonify({
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/autonomous-tasking', methods=['POST'])
def autonomous_tasking():
    """Alias for /api/query to support different client conventions."""
    return handle_query()

@app.route('/api/autonomous-tasking/stream', methods=['POST'])
def autonomous_tasking_stream():
    data = request.json or {}
    query = data.get('query') or data.get('task', '')
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    app.logger.info(f'=== STREAMING QUERY: {query} ================================================')
    additional_data = {k: v for k, v in data.items() if k not in ('query', 'task')}
    if additional_data:
        app.logger.info(f'Additional context: {additional_data}')
    
    def generate():
        last_sent_count = 0
        iteration = 0
        phase_labels = {
            'thinking': '🧠 THINK',
            'planning': '📋 PLAN',
            'executing': '⚙️ EXECUTE',
            'reflecting': '💭 REFLECT'
        }
        
        result_container = {'result': None, 'error': None}
        
        def run_reasoning_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(agent.process_query(query, additional_data))
                result_container['result'] = result
            except Exception as e:
                app.logger.error(f'Streaming error: {str(e)}')
                result_container['error'] = str(e)
            finally:
                loop.close()
        
        # Start reasoning in a separate thread
        reasoning_thread = threading.Thread(target=run_reasoning_thread)
        reasoning_thread.start()
        
        # Stream updates as reasoning progresses
        while reasoning_thread.is_alive():
            # Check for new reasoning steps
            current_history = agent.reasoning_engine.cycle_history
            new_steps = current_history[last_sent_count:]
            
            for step in new_steps:
                step_dict = step.to_dict()
                state = step_dict.get('state', '')
                
                if state == 'planning':
                    label = f'🧠 PLANNING (Cycle {step_dict.get("cycle", 1)})'
                elif state == 'execution':
                    label = f'⚙️ EXECUTION (Cycle {step_dict.get("cycle", 1)})'
                else:
                    label = state.upper()
                
                log_entry = {
                    'type': 'phase',
                    'step': label,
                    'summary': step_dict.get('reasoning', ''),
                    'confidence': step_dict.get('confidence', 0.0),
                    'action': step_dict.get('action_selected', ''),
                    'results': step_dict.get('results', {})
                }
                
                yield f"data: {json.dumps(log_entry)}\n\n"
                last_sent_count += 1
            
            # Small sleep to avoid busy waiting
            time.sleep(0.1)
        
        # Wait for thread to complete
        reasoning_thread.join()
        
        # Get final result
        if result_container['error']:
            yield f"data: {json.dumps({'type': 'error', 'error': result_container['error']})}\n\n"
        else:
            result = result_container['result']
            if result and 'error' in result:
                yield f"data: {json.dumps({'type': 'error', 'error': result['error']})}\n\n"
            else:
                # Send any remaining steps
                current_history = agent.reasoning_engine.cycle_history
                new_steps = current_history[last_sent_count:]
                
                for step in new_steps:
                    step_dict = step.to_dict()
                    state = step_dict.get('state', '')
                    
                    if state == 'planning':
                        label = f'🧠 PLANNING (Cycle {step_dict.get("cycle", 1)})'
                    elif state == 'execution':
                        label = f'⚙️ EXECUTION (Cycle {step_dict.get("cycle", 1)})'
                    else:
                        label = state.upper()
                    
                    log_entry = {
                        'type': 'phase',
                        'step': label,
                        'summary': step_dict.get('reasoning', ''),
                        'confidence': step_dict.get('confidence', 0.0),
                        'action': step_dict.get('action_selected', ''),
                        'results': step_dict.get('results', {})
                    }
                    
                    yield f"data: {json.dumps(log_entry)}\n\n"
                
                # Send completion event
                cycles = result.get("total_cycles", 0)
                yield f"data: {json.dumps({'type': 'complete', 'result': result, 'cycles': cycles})}\n\n"
    
    return Response(stream_with_context(generate()), mimetype='text/event-stream', headers={
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no'
    })

@app.route('/api/task-history', methods=['GET'])
def get_task_history():
    limit = int(request.args.get('limit', 20))
    return jsonify({
        'status': 'success',
        'tasks': agent.task_history[-limit:],
        'total': len(agent.task_history)
    })

@app.route('/api/task-history/<int:task_id>', methods=['GET'])
def get_task_details(task_id):
    for task in agent.task_history:
        if task['id'] == task_id:
            return jsonify({'status': 'success', 'task': task})
    return jsonify({'error': 'Task not found'}), 404

@app.route('/api/task-history/clear', methods=['POST'])
def clear_task_history():
    agent.task_history = []
    return jsonify({'status': 'success', 'message': 'Task history cleared'})

# Server logs written to files for debugging purposes

@app.route('/api/memory/status', methods=['GET'])
def memory_status():
    """Get status of all memory modules (CoALA)."""
    if agent is None:
        return jsonify({'error': 'Agent not initialized'}), 503
    
    return jsonify({
        'status': 'success',
        'memory_modules': {
            'working': {
                'size': agent.working_memory.size(),
                'current_state_keys': list(agent.working_memory.get_current_state().keys())
            },
            'episodic': {
                'size': agent.episodic_memory.size(),
                'statistics': agent.episodic_memory.get_statistics()
            },
            'semantic': {
                'size': agent.semantic_memory.size(),
                'statistics': agent.semantic_memory.get_statistics()
            },
            'procedural': {
                'size': agent.procedural_memory.size(),
                'statistics': agent.procedural_memory.get_statistics()
            }
        },
        'action_space': agent.reasoning_engine.action_space.get_action_summary()
    })

@app.route('/api/memory/clear', methods=['POST'])
def clear_memory():
    """Clear memory modules (for development/testing)."""
    if agent is None:
        return jsonify({'error': 'Agent not initialized'}), 503
    
    data = request.json or {}
    memory_type = data.get('memory_type', 'all')
    
    cleared = []
    
    if memory_type in ['all', 'working']:
        agent.working_memory.clear_all()
        cleared.append('working')
    
    if memory_type in ['all', 'episodic']:
        agent.episodic_memory.clear()
        cleared.append('episodic')
    
    if memory_type in ['all', 'semantic']:
        agent.semantic_memory.clear()
        cleared.append('semantic')
    
    if memory_type in ['all', 'procedural']:
        agent.procedural_memory.clear()
        cleared.append('procedural')
    
    return jsonify({
        'status': 'success',
        'message': f'Cleared memory modules: {", ".join(cleared)}',
        'cleared': cleared
    })

@app.route('/api/managed-satellites')
def get_managed_satellites():
    """Get managed satellites data."""
    try:
        with open('data/managed_satellites.toon', 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except FileNotFoundError:
        return jsonify({'error': 'Managed satellites data not found'}), 404
    except json.JSONDecodeError as e:
        return jsonify({'error': f'Invalid JSON: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
