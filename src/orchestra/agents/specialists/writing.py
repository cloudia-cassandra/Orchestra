from orchestra.agents.specialists.base_specialist import SpecialistAgent

SYSTEM_PROMPT = """You are the Writing Specialist. Given an instruction and any supporting \
material, produce clear, well-structured prose suited to the intended audience. Prefer plain \
language over jargon unless the instruction calls for a technical register."""


class WritingAgent(SpecialistAgent):
    name = "writing"
    domain = "writing"
    system_prompt = SYSTEM_PROMPT
    tools = ["file_read_write"]
