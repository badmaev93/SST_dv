const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  getVersion:      () => ipcRenderer.invoke('get-version'),
  getServerUrl:    () => ipcRenderer.invoke('get-server-url'),
  resetServerUrl:  () => ipcRenderer.invoke('reset-server-url'),
  setServerUrl:    (url) => ipcRenderer.send('set-server-url', url),
});
