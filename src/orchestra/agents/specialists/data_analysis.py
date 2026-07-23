from orchestra.agents.specialists.base_specialist import SpecialistAgent

SYSTEM_PROMPT = """You are the Data Analysis Specialist. Given an instruction, work through the \
data reasoning or computation required, showing your method, and state your result clearly. \
Call out assumptions you had to make about the data."""


class DataAnalysisAgent(SpecialistAgent):
    name = "data_analysis"
    domain = "data_analysis"
    system_prompt = SYSTEM_PROMPT
    # Intended tools: sql_query, python_sandbox, dataframe_ops (bound in a later ticket)
    tools = ["sql_query", "python_sandbox"]
