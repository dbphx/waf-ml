import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "random_forest"))
from predict import HTTPAttackPredictor

p = HTTPAttackPredictor(os.path.join(PROJECT_ROOT, "models/random_forest"), use_onnx=True)

q2 = (
    "response-content-disposition=inline%3B%20filename%2A%3DUTF-8%27%279a0505e796d842fdb7a4980ead8c4bf3"
    "&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=9wnmzzHYVGwlnTscRdzx%2F20260519%2F%2Fs3%2Faws4_request"
    "&X-Amz-Date=20260519T112548Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host"
    "&X-Amz-Signature=cdf2aeefc1bf8bd6d456ede48d0a809d4f051e52eece84bacb322945a8c50a1b"
)
q3 = "expand=issue_reactions,issue_attachments,issue_link,parent"
body4 = (
    '{"description_html":"<p class=\\"editor-paragraph-block\\" data-id=\\"a271033a-c405-4c50-abd2-ab36980e3282\\">'
    '<span>%</span>3CxssBypass<span>/</span>onpointermove<span>=(</span>confirm<span>)(1)%</span>3EMoveMouseHere</p>"}'
)

cases = [
    ("hit1", {"method": "GET", "path": "/insky/projects/bce0a79c-90d2-4558-9084-945ad6acbdae/issues/", "query": "", "headers": "", "body": ""}),
    ("hit2", {"method": "GET", "path": "/uploads/628668cae9db4d51b4edf55214b73fca-Facebook Image.jpg", "query": q2, "headers": "", "body": ""}),
    ("hit3", {"method": "GET", "path": "/api/workspaces/insky/projects/bce0a79c-90d2-4558-9084-945ad6acbdae/issues/9906eeae-3678-40e2-9869-64bc8b84c7c5/", "query": q3, "headers": "", "body": ""}),
    ("hit4 path", {"method": "PATCH", "path": "/api/workspaces/insky/projects/bce0a79c-90d2-4558-9084-945ad6acbdae/issues/9906eeae-3678-40e2-9869-64bc8b84c7c5/", "query": "", "headers": "", "body": ""}),
    ("hit4 body", {"method": "PATCH", "path": "/api/workspaces/insky/projects/bce0a79c-90d2-4558-9084-945ad6acbdae/issues/9906eeae-3678-40e2-9869-64bc8b84c7c5/", "query": "", "headers": "", "body": body4}),
]

for name, req in cases:
    r = p.predict_components(req)
    print(f"{name}: {r['prediction']} conf={r['confidence']} decisive={r['decisive_component']}")
    for k, v in sorted(r["components"].items(), key=lambda x: -x[1]["probability"]):
        print(f"  {k} {v['probability']:.4f} {v['prediction']}")
