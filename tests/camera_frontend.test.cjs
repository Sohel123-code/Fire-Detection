const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function setup(getUserMedia) {
  const elements = new Map(), messages = [], listeners = {}, timers = new Map();
  let timerId = 0, released = 0;
  const track = {stop() { released++; }, addEventListener() {}};
  const stream = {getTracks: () => [track], getVideoTracks: () => [track]};
  const element = id => {
    if (!elements.has(id)) elements.set(id, {
      value: 'user', readyState: 2, videoWidth: 640, videoHeight: 480,
      play: async () => {}, addEventListener(type, fn) { this[type] = fn; },
      removeAttribute(key) { delete this[key]; },
      getContext: () => ({drawImage() {}}), toDataURL: () => 'data:image/jpeg;base64,dGVzdA=='
    });
    return elements.get(id);
  };
  const parent = {postMessage(message) { messages.push(message); }};
  const window = {parent, isSecureContext: true, addEventListener(type, fn) { listeners[type] = fn; }};
  vm.runInNewContext(fs.readFileSync('camera_frontend/camera.js', 'utf8'), {
    window, document: {getElementById: element, documentElement: {scrollHeight: 500}, body: {}},
    navigator: {mediaDevices: {getUserMedia: getUserMedia || (async () => stream)}},
    ResizeObserver: class {observe() {}},
    setTimeout(fn, delay) {timers.set(++timerId, {fn, delay}); return timerId;},
    clearTimeout(id) {timers.delete(id);}
  });
  return {
    element, messages, timers, listeners, stream, get released() {return released;},
    frames: () => messages.filter(m => m.value?.kind === 'frame'),
    reply(reply) {listeners.message({source: parent, data: {type: 'streamlit:render', args: {reply}}});},
    tick(delay) {for (const [id, timer] of timers) if (timer.delay === delay) {timers.delete(id); timer.fn(); break;}}
  };
}

test('continuous capture waits for each matching prediction and stops camera tracks', async () => {
  const app = setup();
  await app.element('start').click();
  assert.equal(app.frames().length, 1);
  for (let i = 0; i < 30; i++) {
    const frame = app.frames().at(-1).value;
    app.reply({id: 'wrong-session', image: 'wrong'});
    assert.equal(app.frames().length, i + 1);
    app.reply({id: frame.id, fire: 0, smoke: 1, image: 'data:image/jpeg;base64,result'});
    app.tick(100);
    assert.equal(app.frames().length, i + 2);
  }
  app.element('stop').click();
  assert.equal(app.released, 1);
  assert.equal(app.timers.size, 0);
  assert.equal(app.element('video').srcObject, null);
  const count = app.frames().length;
  app.reply({id: app.frames().at(-1).value.id, fire: 0, smoke: 0});
  app.tick(100);
  assert.equal(app.frames().length, count);
});

test('permission denial is visible and Start remains available', async () => {
  const app = setup(async () => {throw {name: 'NotAllowedError'};});
  await app.element('start').click();
  assert.match(app.element('status').textContent, /permission denied/);
  assert.equal(app.element('start').disabled, false);
  assert.equal(app.frames().length, 0);
});

test('stop while permission is pending releases the late camera stream', async () => {
  let resolve;
  const app = setup(() => new Promise(r => {resolve = r;}));
  const opening = app.element('start').click();
  app.element('stop').click();
  resolve(app.stream);
  await opening;
  assert.equal(app.released, 1);
  assert.equal(app.frames().length, 0);
});

test('inference error is acknowledged and the next frame can recover', async () => {
  const app = setup();
  await app.element('start').click();
  app.reply({id: app.frames()[0].value.id, error: 'model unavailable'});
  assert.match(app.element('status').textContent, /model unavailable/);
  app.tick(1000);
  assert.equal(app.frames().length, 2);
});

test('unmount releases the physical camera and cancels pending capture', async () => {
  const app = setup();
  await app.element('start').click();
  app.listeners.pagehide();
  assert.equal(app.released, 1);
  assert.equal(app.timers.size, 0);
});
