/* Streamlit component protocol; all assets and frames stay on the app connection. */
(() => {
  const $ = (id) => document.getElementById(id);
  const video = $('video'), prediction = $('prediction'), canvas = $('capture');
  let stream = null, active = false, session = '', sequence = 0;
  let pending = null, timer = null, generation = 0, initialized = false;
  const send = (type, fields) => window.parent.postMessage({isStreamlitMessage: true, type, ...fields}, '*');
  const emit = (value) => send('streamlit:setComponentValue', {value, dataType: 'json'});
  const resize = () => send('streamlit:setFrameHeight', {height: document.documentElement.scrollHeight});
  function status(text, error = false) {
    $('status').textContent = text;
    $('status').className = error ? 'error' : '';
    resize();
  }
  function release() {
    active = false;
    generation++;
    clearTimeout(timer);
    pending = null;
    if (stream) stream.getTracks().forEach(track => track.stop());
    stream = null;
    video.srcObject = null;
    prediction.removeAttribute('src');
    $('start').disabled = false;
    $('stop').disabled = true;
    $('facing').disabled = false;
  }
  function stop() {
    release();
    status('Camera stopped.');
    emit({kind: 'stopped', session});
  }
  function capture() {
    if (!active || pending) return;
    if (video.readyState < 2 || !video.videoWidth) {
      timer = setTimeout(capture, 200);
      return;
    }
    const scale = Math.min(1, 640 / Math.max(video.videoWidth, video.videoHeight));
    canvas.width = Math.max(2, Math.round(video.videoWidth * scale));
    canvas.height = Math.max(2, Math.round(video.videoHeight * scale));
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
    pending = `${session}:${++sequence}`;
    emit({kind: 'frame', session, id: pending, image: canvas.toDataURL('image/jpeg', 0.75)});
    // Exactly one request in flight. A timeout offers recovery without a growing queue.
    timer = setTimeout(() => {
      if (active && pending) {
        status('Prediction response timed out. Stop and start the camera to retry.', true);
      }
    }, 30000);
  }
  async function start() {
    release();
    const attempt = generation;
    session = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    sequence = 0;
    $('start').disabled = true;
    $('stop').disabled = false;
    status('Requesting camera access…');
    try {
      if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
        throw new Error('Open the app directly at its HTTPS address to allow camera access.');
      }
      const acquired = await navigator.mediaDevices.getUserMedia({
        video: {facingMode: {ideal: $('facing').value}, width: {ideal: 640}, height: {ideal: 480}},
        audio: false
      });
      if (attempt !== generation) { acquired.getTracks().forEach(track => track.stop()); return; }
      stream = acquired;
      video.srcObject = stream;
      await video.play();
      if (attempt !== generation) return;
      active = true;
      $('facing').disabled = true;
      stream.getVideoTracks()[0].addEventListener('ended', stop);
      status('Camera live · waiting for the first prediction…');
      capture();
    } catch (error) {
      if (attempt !== generation) return;
      const messages = {
        NotAllowedError: 'Camera permission denied. Allow camera access in your browser site settings, then click Start camera.',
        NotFoundError: 'No camera found. Connect a camera and try again.',
        NotReadableError: 'Camera is busy or unavailable. Close other camera apps and try again.',
        OverconstrainedError: 'This camera cannot use the requested settings. Select another camera and try again.'
      };
      const message = messages[error.name] || error.message || 'Could not start the camera.';
      release();
      status(message, true);
      emit({kind: 'error', session, message});
    }
  }
  window.addEventListener('message', event => {
    if (event.source !== window.parent || event.data?.type !== 'streamlit:render') return;
    const reply = event.data.args?.reply;
    if (!initialized) { initialized = true; resize(); }
    if (!active || !pending || !reply || reply.id !== pending) return;
    clearTimeout(timer);
    pending = null;
    if (reply.error) {
      prediction.removeAttribute('src');
      status(`Prediction failed: ${reply.error}`, true);
    } else {
      prediction.src = reply.image;
      status(`Predicting continuously · Fire: ${reply.fire} · Smoke: ${reply.smoke}`);
    }
    timer = setTimeout(capture, reply.error ? 1000 : 100);
  });
  $('start').addEventListener('click', start);
  $('stop').addEventListener('click', stop);
  window.addEventListener('pagehide', release);
  window.addEventListener('resize', resize);
  new ResizeObserver(resize).observe(document.body);
  send('streamlit:componentReady', {apiVersion: 1});
  resize();
})();
