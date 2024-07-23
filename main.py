"""Root module in quiz project."""

from typing import Literal
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(
  model="gemini-1.5-flash"
) # type: ignore

Difficulty = Literal["easy", "medium", "hard"]

class QuizModel(BaseModel):
  title: str
  num_questions: int
  difficulty: Difficulty

Option = Literal['a', 'b', 'c', 'd']

class Question(BaseModel):
  id: int
  question: str
  options: dict[Option, str]
  correct: str

class QuizOutput(BaseModel):
  total_questions: int
  questions: list[Question]

app = FastAPI()

@app.get("/healthcheck")
def root():
  return {
    "message": "success"
  }

@app.get("/quiz")
async def generate_quiz(quiz_body: QuizModel):
  try:
    output = await get_quiz_from_ai(quiz_body)
    return {
      "message": "success",
      "body": output
    }
  except Exception as err:
    print(err)
    return {
      "message": "error"
    }


async def get_quiz_from_ai(quiz_config: QuizModel):
  template="""
  Quiz title: {title}
  You are an expert MCQ maker. Given the above title, it is your job to
  create a quiz  of {number} multiple choice questions. Keep the difficulty {difficulty}.
  Make sure the questions are not repeated and check all the questions to be conforming the text as well.

  Make sure to format the response in the below format
  {format_instruction}
  """

  prompt = PromptTemplate.from_template(template=template)
  parser = JsonOutputParser(pydantic_object=QuizOutput)

  chain = prompt | model | parser

  output = chain.invoke({ 
    "title": quiz_config.title, 
    "number": quiz_config.num_questions, 
    "difficulty": quiz_config.difficulty, 
    "format_instruction": parser.get_format_instructions() 
  })

  return output