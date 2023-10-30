from bokeh.embed import server_document
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse
import user_agent


app = FastAPI()
templates = Jinja2Templates(directory="./APIs/templates")

@app.get("/test_hello")
async def root():
    return {"message": "Hello World"}


@app.get("/")
async def bkapp_page(request: Request):
    script = server_document('http://127.0.0.1:5000')
    #print(script)
    return templates.TemplateResponse("new_base.html", {"request": request, "script": script})





