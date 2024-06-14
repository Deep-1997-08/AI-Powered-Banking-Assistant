from langchain import AgenticWorkflow

def handle_query(query):
    workflow = AgenticWorkflow()
    result = workflow.execute(query)
    return result
