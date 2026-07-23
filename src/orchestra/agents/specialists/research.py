from orchestra.agents.specialists.base_specialist import SpecialistAgent

SYSTEM_PROMPT = """You are the Research Specialist. Given an instruction, gather and summarize \
the relevant facts needed to complete it. Be precise, cite sources when you reference specific \
claims, and flag anything you're uncertain about rather than guessing."""


class ResearchAgent(SpecialistAgent):
    name = "research"
    domain = "research"
    system_prompt = SYSTEM_PROMPT
    tools = ["web_search", "api_call"]
