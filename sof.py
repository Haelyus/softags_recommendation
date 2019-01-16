from flask import Flask, render_template, request, flash
from forms import SofForm
import calendar
import fctsof as fs

app = Flask(__name__)
app.secret_key = 'development key'


@app.route('/sof', methods = ['GET', 'POST'])
def sof():
    form = SofForm()
    
    if request.method == 'POST':
        if form.validate() == False:
            flash('All fields are required.')
            return render_template('sof.html', form = form)
        else:
            title = form.title.data
            body = form.body.data
            #question = fs.cleaned_question(title, body)
            question = fs.tags_recommendation(title, body)
            form.title.data = str(title)
            form.body.data = str(body)
            form.tags.data = str(question)
            return render_template('success.html', form = form)
    elif request.method == 'GET':
        return render_template('sof.html', form = form)

@app.errorhandler(404)
def page_not_found(error):
    return render_template('errors/404.html'), 404		
		
if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True, port = 8080)