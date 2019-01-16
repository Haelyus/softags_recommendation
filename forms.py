from flask_wtf import Form
from wtforms import TextField, TextAreaField, SubmitField

from wtforms import validators, ValidationError

import numpy as np
import pandas as pd


class SofForm(Form):
	   
    title = TextField("Title",[validators.InputRequired("Please enter the title of the question.")])
    body = TextAreaField("Body",[validators.InputRequired("Please enter the body of the question.")])
    tags = TextAreaField("",[validators.InputRequired("")])
    
    submit = SubmitField("Send")