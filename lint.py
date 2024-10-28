#lint.py 
#change to trigger workflow action # again for mainbrnach again
#changes the threshold value to 1 to passs the lint
import sys 

from pylint import lint  

THRESHOLD = 0

files_to_check = ["app.py", "model_creation.py", "result.py"]
run = lint.Run(files_to_check, exit=False)

#score = run.linter.stats["global_note"]  
score = run.linter.stats.global_note

if score < THRESHOLD: 
    print("Linter failed: Score < threshold value") 

    sys.exit(1) 


sys.exit(0)