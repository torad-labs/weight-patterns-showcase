import {defineConfig} from 'vite';
import {resolve} from 'node:path';
import {weightApi} from './tools/weight-api.ts';
export default defineConfig({plugins:[weightApi()],build:{rollupOptions:{input:{replay:resolve(import.meta.dirname,'index.html'),teaching:resolve(import.meta.dirname,'teaching.html')}}}});
