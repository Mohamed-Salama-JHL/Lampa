from bokeh.embed import server_document
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse


app = FastAPI()
templates = Jinja2Templates(directory="./APIs/templates")

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/dashboard")
async def bkapp_page(request: Request):
    script = server_document('http://127.0.0.1:5000/app')
    #print(script)
    return templates.TemplateResponse("base.html", {"request": request, "script": script})


@app.get("/get-screen-resolution")
async def get_screen_resolution():
    html_script = """
    <script>
    var resolution = {
        width: window.innerWidth || document.documentElement.clientWidth || document.body.clientWidth,
        height: window.innerHeight || document.documentElement.clientHeight || document.body.clientHeight
    };
    document.body.innerText = JSON.stringify(resolution);
    </script>
    """
    return HTMLResponse(content=html_script)





