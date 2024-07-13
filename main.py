import os
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
from typing import List
from dotenv import load_dotenv

from educhain import qna_engine, content_engine

# Load environment variables
load_dotenv()

app = FastAPI()

@app.get("/", status_code=status.HTTP_200_OK)
def root():
    return {"message": "Server is running"}

# Define input models


class MCQRequest(BaseModel):
    grade: str
    subject: str
    topic: str
    subtopic: str
    isNcert: bool = False
    numberOfQuestions: int
    customInstructions: str = ""


class LessonPlanRequest(BaseModel):
    subject: str
    topic: str  # Added topic parameter
    grade: int
    duration: int  # Duration in minutes
    custom_instructions: str


class NCERTLessonPlan(BaseModel):
    subject: str
    topic: str
    grade: int
    duration: str
    objectives: List[str]
    prerequisites: List[str]
    introduction: str
    content_outline: List[str]
    activities: List[str]
    assessment: str
    conclusion: str
    resources: List[str]
    timeline: List[str]

# MCQ generation endpoint


@app.post("/generate-mcq", status_code=status.HTTP_200_OK)
async def api_generate_mcq_questions(request: MCQRequest):
    try:
        llama3_groq = ChatOpenAI(model = "llama3-70b-8192",
                                openai_api_base = "https://api.groq.com/openai/v1",
                                openai_api_key = os.getenv("GROQ_API_KEY"))
        custom_ncert_template = """
        Generate {num} multiple-choice question (MCQ) based on the given topic and level.
        Provide the question, four answer options, and the correct answer.
        Topic: {topic}
        Subtopic: {subtopic}
        Subject: {subject}
        Grade: {grade}
        """

        result = qna_engine.generate_mcq(
            topic=request.topic,
            num=request.numberOfQuestions,
            subject=request.subject,
            grade=request.grade,
            custom_instructions=custom_ncert_template + request.customInstructions,
            is_ncert=request.isNcert,
            prompt_template=custom_ncert_template,
            subtopic=request.subtopic,
            llm = llama3_groq
        )
        return result
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
