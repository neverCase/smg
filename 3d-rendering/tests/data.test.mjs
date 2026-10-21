import test from 'node:test';
import assert from 'node:assert/strict';
import { normalizePayload, summarize, relativeLoads, heatColor, validateEndpoint, demoPayload } from '../src/data.js';

const worker=(overrides={})=>({id:'a',url:'http://worker-a:9000',is_healthy:true,status:'ready',model_id:'Qwen',load:26,...overrides});
test('preserves the distinction between SMG in-flight and sampled engine counts',()=>{
  const {workers}=normalizePayload({workers:[worker({engine_load:{loads:[{num_running_reqs:22,num_waiting_reqs:3,token_usage:.1649}]}})]});
  assert.equal(workers[0].load,26);assert.equal(workers[0].running,22);assert.equal(workers[0].waiting,3);
  assert.equal(workers[0].capacity,null);assert.equal(workers[0].engineTimestamp,null);
});
test('unknown metrics remain null while real zeros stay zero',()=>{
  const {workers}=normalizePayload({workers:[worker(),worker({id:'b',load:0,engine_load:{loads:[{num_running_reqs:0,num_waiting_reqs:0,token_usage:0,max_total_num_tokens:0}]}})]});
  assert.equal(workers[0].running,null);assert.equal(workers[1].running,0);assert.equal(workers[1].kv,0);assert.equal(workers[1].capacity,null);
  assert.deepEqual(summarize(workers).running,{value:0,partial:true});
});
test('groups every supported model without double-counting a worker in the cluster',()=>{
  const raw=worker({models:[{id:'Qwen'},{id:'Llama'},{id:'Qwen'}]});
  const data=normalizePayload({workers:[raw,raw]});assert.equal(data.workers.length,1);assert.equal(data.groups.length,2);assert.equal(summarize(data.workers).inflight.value,26);
});
test('relative heat is calculated within the group, excludes offline nodes, and handles ties',()=>{
  const {workers}=normalizePayload({workers:[worker({id:'a',load:10}),worker({id:'b',load:20}),worker({id:'c',load:100,is_healthy:false})]});
  const heat=relativeLoads(workers);assert.equal(heat.get('a').heat,0);assert.equal(heat.get('b').heat,1);assert.equal(heat.get('c').heat,null);assert.equal(heat.get('b').max,20);
  assert.equal(relativeLoads([workers[1]]).get('b').heat,0);assert.equal(heatColor(0),'#70dce6');assert.equal(heatColor(1),'#ff656c');
});
test('multi-rank counts sum, usage averages, incomplete ranks are not silently zeroed',()=>{
  const data=normalizePayload({workers:[worker({engine_load:{loads:[{num_running_reqs:2,num_waiting_reqs:3,token_usage:.2},{num_running_reqs:7,token_usage:.8}]}})]});
  assert.equal(data.workers[0].running,9);assert.equal(data.workers[0].waiting,null);assert.equal(data.workers[0].kv,.5);
});
test('rejects malformed payloads and unsafe endpoint protocols',()=>{
  assert.throws(()=>normalizePayload({error:'bad'}));assert.throws(()=>normalizePayload({workers:[null]}));assert.throws(()=>validateEndpoint('javascript:alert(1)'));assert.throws(()=>validateEndpoint('http://user:pass@localhost/workers'));
  assert.equal(validateEndpoint('http://localhost:3000/workers'),'http://localhost:3000/workers');
});
test('demo is a valid multi-model topology and empty live fleets are valid',()=>{
  assert.equal(normalizePayload(demoPayload()).groups.length,4);assert.deepEqual(summarize([]).waiting,{value:0,partial:false});assert.equal(normalizePayload({workers:[]}).workers.length,0);
});
