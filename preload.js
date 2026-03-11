const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('proctor', {
  sendEvent: (type, data) => {
    ipcRenderer.send('proctor-event', { type, ...data, timestamp: Date.now() });
  }
});