import {test,expect} from 'bun:test';
import {decisionAlternatives,readTraceStudy} from './trace-model';
const decision=(id:number)=>({step:0,selected:id,piece:String(id),top:[{id,piece:String(id),logit:12}]});
const fixture=()=>({schema:'connected-weight-trace-v1',stages:['input'],cases:[{fidelity:{tokens:true,fullScoreHashes:true},boundary:{common:0,divergence:0,eligible:1},original:{tokens:[1,2],steps:[decision(1),decision(2)]},edited:{tokens:[3,2],steps:[decision(3),decision(2)]},predictions:[{step:0,measures:[{name:'input',absolute:0,relative:0}]}]}]});
test('only incoming-context-matched decisions enter the connected comparison',()=>{expect(readTraceStudy(fixture()).cases[0].boundary.eligible).toBe(1);const bad=fixture();bad.cases[0].boundary.eligible=2;bad.cases[0].predictions.push({...bad.cases[0].predictions[0],step:1});expect(()=>readTraceStudy(bad)).toThrow('shared input history');});
test('missing top-eight alternatives remain unknown, never zero',()=>{const rows=decisionAlternatives({step:0,position:0,inputCoefficient:0,measures:[],original:decision(1),edited:decision(3)});expect(rows[0].edited).toBeNull();expect(rows[1].original).toBeNull();});
