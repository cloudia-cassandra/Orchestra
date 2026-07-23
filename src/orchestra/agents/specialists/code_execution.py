from orchestra.agents.specialists.base_specialist import SpecialistAgent

SYSTEM_PROMPT = """You are the Code Execution Specialist. Given an instruction, write correct, \
minimal code that accomplishes it, explain what it does, and note any risks (side effects, \
destructive operations, external calls) before it would be run."""


class CodeExecutionAgent(SpecialistAgent):
    name = "code_execution"
    domain = "code_execution"
    system_prompt = SYSTEM_PROMPT
    # Intended tools: sandboxed_shell, file_read_write (bound in a later ticket)
    tools = ["sandboxed_shell"]
