#lint.py 

import sys 

from pylint import lint  

THRESHOLD = 9 

files_to_check = ["app.py", "model_creation.py", "result.py"]
run = lint.Run(files_to_check, exit=False)

#score = run.linter.stats["global_note"]  
score = run.linter.stats.global_note

if score < THRESHOLD: 
    print("Linter failed: Score < threshold value") 

    sys.exit(1) 


sys.exit(0)