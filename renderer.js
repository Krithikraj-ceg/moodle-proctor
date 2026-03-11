

const video = document.getElementById('video');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');

const faceStatusEl = document.getElementById('face-status');
const faceDotEl = document.getElementById('face-dot');
const faceNoteEl = document.getElementById('face-note');
const gazeValueEl = document.getElementById('gaze-value');
const eyeValueEl  = document.getElementById('eye-value');

function sendProctorEvent(type, data = {}) {
  if (window.proctor && typeof window.proctor.sendEvent === 'function') {
    window.proctor.sendEvent(type, data);
  }
}

/* ---------------- LANDMARK INDICES ---------------- */

const LEFT_EYE_OUTER = 33;
const LEFT_EYE_INNER = 133;
const RIGHT_EYE_OUTER = 362;
const RIGHT_EYE_INNER = 263;

const LEFT_IRIS = [468,469,470,471];
const RIGHT_IRIS = [473,474,475,476];

const LEFT_EYE_TOP = 159;
const LEFT_EYE_BOTTOM = 145;
const RIGHT_EYE_TOP = 386;
const RIGHT_EYE_BOTTOM = 374;

/* ---------------- VARIABLES ---------------- */

let faceMesh = null;
let camera = null;

let lastFaceSeenTime = 0;
let lastGazeLabel = "center";
let lastEyeLabel = "open";

let gazeHistory = [];
let offScreenStart = null;

/* ---------------- CAMERA ---------------- */

async function initCamera(){

  const stream = await navigator.mediaDevices.getUserMedia({
    video:{width:640,height:480},
    audio:false
  });

  video.srcObject = stream;

  return new Promise(resolve=>{
    video.onloadedmetadata = ()=>{
      video.play();
      resolve();
    }
  });
}

/* ---------------- MEDIAPIPE ---------------- */

async function initModel(){

  faceMesh = new FaceMesh({
    locateFile:(file)=>`https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
  });

  faceMesh.setOptions({
    maxNumFaces:1,
    refineLandmarks:true,
    minDetectionConfidence:0.5,
    minTrackingConfidence:0.5
  });

  faceMesh.onResults(handleResults);
}

/* ---------------- HELPERS ---------------- */

function getCenter(landmarks,indices){

  let x=0,y=0;

  indices.forEach(i=>{
    x+=landmarks[i].x;
    y+=landmarks[i].y;
  });

  return{
    x:x/indices.length,
    y:y/indices.length
  }
}

/* ---------------- EYE RATIO ---------------- */

function eyeAspectRatio(landmarks,top,bottom,outer,inner){

  const topP = landmarks[top];
  const bottomP = landmarks[bottom];
  const outerP = landmarks[outer];
  const innerP = landmarks[inner];

  const v = Math.hypot(topP.x-bottomP.x,topP.y-bottomP.y);
  const h = Math.hypot(outerP.x-innerP.x,outerP.y-innerP.y);

  return v/h;
}

/* ---------------- GAZE DETECTION ---------------- */

function getGazeDirection(landmarks){
   console.log("hello");
  const leftOuter = landmarks[LEFT_EYE_OUTER];
  const leftInner = landmarks[LEFT_EYE_INNER];

  const rightOuter = landmarks[RIGHT_EYE_OUTER];
  const rightInner = landmarks[RIGHT_EYE_INNER];

  const leftIris = getCenter(landmarks, LEFT_IRIS);
  const rightIris = getCenter(landmarks, RIGHT_IRIS);

  function horizontalRatio(outer, inner, iris){
    const total = Math.abs(inner.x - outer.x);
    const pos = Math.abs(iris.x - outer.x);
    return pos / total;
  }

  const leftRatio = horizontalRatio(leftOuter,leftInner,leftIris);
  const rightRatio = horizontalRatio(rightOuter,rightInner,rightIris);

  const avgRatio = (leftRatio + rightRatio) / 2;

  // vertical normalization
  const top = landmarks[LEFT_EYE_TOP];
  const bottom = landmarks[LEFT_EYE_BOTTOM];

  const vertRatio = (leftIris.y - top.y) / (bottom.y - top.y);
   
   console.log("Horizontal:", avgRatio.toFixed(2),
            "Vertical:", vertRatio.toFixed(2));
  if (avgRatio < 0.40) return "right";
  if (avgRatio > 0.60) return "left";

  if (vertRatio < 0.28) return "up";
  if (vertRatio > 0.62) return "down";

  return "center";
}
/* ---------------- SMOOTHING ---------------- */

function smoothDirection(dir){

  gazeHistory.push(dir);

  if(gazeHistory.length>5)
    gazeHistory.shift();

  const counts={};

  gazeHistory.forEach(d=>{
    counts[d]=(counts[d]||0)+1
  });

  return Object.keys(counts)
  .reduce((a,b)=>counts[a]>counts[b]?a:b);
}

/* ---------------- EYE STATE ---------------- */

function getEyeOpenState(landmarks){

  const leftEAR = eyeAspectRatio(
    landmarks,
    LEFT_EYE_TOP,
    LEFT_EYE_BOTTOM,
    LEFT_EYE_OUTER,
    LEFT_EYE_INNER
  );

  const rightEAR = eyeAspectRatio(
    landmarks,
    RIGHT_EYE_TOP,
    RIGHT_EYE_BOTTOM,
    RIGHT_EYE_OUTER,
    RIGHT_EYE_INNER
  );

  const avgEAR = (leftEAR+rightEAR)/2;

  return avgEAR < 0.21 ? "closed" : "open";
}

/* ---------------- UI ---------------- */

function updateStatusUI(face,gaze,eye){

  if(!face){

    faceStatusEl.textContent="Face not detected";
    gazeValueEl.textContent="-";
    eyeValueEl.textContent="-";

    return;
  }

  faceStatusEl.textContent="Face detected";

  gazeValueEl.textContent=gaze.toUpperCase();
  eyeValueEl.textContent=eye==="open"?"Eyes open":"Eyes closed";
}

/* ---------------- DRAW ---------------- */

function drawOverlay(width,height,landmarks){

  canvas.width=width;
  canvas.height=height;

  ctx.clearRect(0,0,width,height);

  ctx.save();
  ctx.scale(-1,1);
  ctx.translate(-width,0);

  landmarks.forEach(pt=>{
    ctx.beginPath();
    ctx.arc(pt.x*width,pt.y*height,1.2,0,Math.PI*2);
    ctx.fillStyle="cyan";
    ctx.fill();
  });

  ctx.restore();
}

/* ---------------- RESULTS ---------------- */

function handleResults(results){

  const faces = results.multiFaceLandmarks || [];
  const now = performance.now();

  if(faces.length>0){

    const landmarks = faces[0];

    lastFaceSeenTime = now;

    const rawGaze = getGazeDirection(landmarks);
    const gaze = smoothDirection(rawGaze);

    const eye = getEyeOpenState(landmarks);

    if(gaze !== lastGazeLabel){

      lastGazeLabel = gaze;
      sendProctorEvent("gaze_change",{gaze});
    }

    if(eye !== lastEyeLabel){

      lastEyeLabel = eye;
      sendProctorEvent("eye_state_change",{eye});
    }

    /* suspicious gaze */

    if(gaze !== "center"){

      if(!offScreenStart)
        offScreenStart = now;

      if(now - offScreenStart > 2000){

        sendProctorEvent("suspicious_gaze",{direction:gaze});
      }

    }else{

      offScreenStart=null;
    }

    updateStatusUI(true,gaze,eye);

    if(video.videoWidth)
      drawOverlay(video.videoWidth,video.videoHeight,landmarks);

  }else{

    updateStatusUI(false,lastGazeLabel,lastEyeLabel);

    if(now-lastFaceSeenTime>3000){

      sendProctorEvent("face_missing",{});
    }

    ctx.clearRect(0,0,canvas.width,canvas.height);
  }
}

/* ---------------- MAIN ---------------- */

(async function(){

  await initCamera();

  await initModel();

  camera = new Camera(video,{
    onFrame: async ()=>{
      if(faceMesh)
        await faceMesh.send({image:video});
    },
    width:640,
    height:480
  });

  camera.start();

})();