"""Generate a tailored cover letter paragraph given retrieved context and a job description."""

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

_PROMPT = ChatPromptTemplate.from_template("""
You are helping Dana write a tailored cover letter paragraph.

Her relevant experience:
{context}

Job description:
{job_description}

Write one focused paragraph connecting her experience to this role.
Do not invent anything not in the context.
Keep her tone: direct, specific, no corporate filler.
""")


def generate(context: str, job_description: str) -> str:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    chain = _PROMPT | llm
    response = chain.invoke({
        "context": context,
        "job_description": job_description,
    })
    return response.content
