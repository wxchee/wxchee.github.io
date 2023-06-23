(function(){"use strict";var e={84:function(e,t,a){var l=a(9242),n=a(3396),i=a(7139);const o={class:"drum-web-app"},r={key:0,class:"loader"},s={key:1,class:"loader__visual"},c={class:"navbar"},u=["onClick"],d={class:"view-container"};function p(e,t,a,p,v,g){const m=(0,n.up)("DemoView"),h=(0,n.up)("DataCollectView"),w=(0,n.up)("DevicePanel");return(0,n.wg)(),(0,n.iD)("div",o,[(0,n.Wm)(l.uT,null,{default:(0,n.w5)((()=>[p.audioReady?(0,n.kq)("",!0):((0,n.wg)(),(0,n.iD)("div",r,[p.isAudioLoading?(0,n.kq)("",!0):((0,n.wg)(),(0,n.iD)("div",{key:0,class:"loader__button",onClick:t[0]||(t[0]=(...e)=>p.loadAudioDatas&&p.loadAudioDatas(...e))},"Load Virtual Drum")),p.isAudioLoading?((0,n.wg)(),(0,n.iD)("div",s,"Loading...")):(0,n.kq)("",!0)]))])),_:1}),(0,n._)("div",c,[((0,n.wg)(!0),(0,n.iD)(n.HY,null,(0,n.Ko)(p.VIEWS,((e,t)=>((0,n.wg)(),(0,n.iD)("div",{key:t,class:(0,i.C_)(p.navbarOptionClass(e)),onClick:()=>p.setView(e)},(0,i.zw)(e.label),11,u)))),128))]),(0,n._)("div",d,[p.view.index===p.VIEWS[0].index?((0,n.wg)(),(0,n.j4)(m,{key:0})):((0,n.wg)(),(0,n.j4)(h,{key:1}))]),(0,n.Wm)(w)])}var v=a(4870);const g=["acoustic","analog","latin"],m=(0,v.qj)({0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0}),h=(0,v.iH)(g[0]),w=[],f={q:0,w:1,a:2,s:3,e:4,r:5,d:6,f:7};Object.keys(f).forEach((e=>{f[e.toUpperCase()]=f[e]}));const _=()=>{window.addEventListener("keypress",(e=>{(f[e.key]||0===f[e.key])&&k(f[e.key])}))},k=e=>{E(h.value,e),y(e)},y=e=>{m[e]=1,w[e]&&(cancelAnimationFrame(w[e]),w[e]=null),w[e]=()=>{m[e]-=.1,m[e]<=0?(cancelAnimationFrame(w[e]),w[e]=null,m[e]=0):requestAnimationFrame(w[e])},requestAnimationFrame(w[e])},C=(0,v.iH)(!1),D=(0,v.iH)(!1);let b=null;const x={acoustic:["0-crash-1.wav","1-tom-1.wav","2-hihat.wav","3-snare-1.wav","4-tom-2.wav","5-crash-2.wav","6-kick-1.wav","7-snare-2.wav"],analog:["0-crash-1.wav","1-clap.wav","2-hihat-1.wav","3-snare-1.wav","4-conga.wav","5-crash-2.wav","6-kick.wav","7-snare-2.wav"],latin:["0-Guiro.wav","1-Conga-4.wav","2-Shaker.wav","3-Conga.wav","4-Bongo.wav","5-CoffeeCup.wav","6-Timbale-2.wav","7-WoodBlock.wav"]},S=e=>fetch(e).then((e=>e.arrayBuffer())).then((e=>b.decodeAudioData(e))),T={acoustic:null,analog:null,latin:null},F=()=>{D.value=!0,b=new AudioContext,Promise.all([...x["acoustic"].map((e=>S(`audios/acoustic/${e}`))),...x["analog"].map((e=>S(`audios/analog/${e}`))),...x["latin"].map((e=>S(`audios/latin/${e}`)))]).then((e=>{T["acoustic"]=e.slice(0,8),T["analog"]=e.slice(8,16),T["latin"]=e.slice(16,24),_(),setTimeout((()=>{C.value=!0,D.value=!1,console.log("audio data loaded")}),300)}))},E=(e,t)=>{const a=b.createBufferSource();a.buffer=T[e][t],a.connect(b.destination),a.start(0)},V={class:"demo-view"},A={class:"option"},M={class:"option-content"},B={class:"drum-type"};function H(e,t,a,l,i,o){const r=(0,n.up)("DrumSettings"),s=(0,n.up)("ToggleButton"),c=(0,n.up)("DrumVisual"),u=(0,n.up)("RadioButton");return(0,n.wg)(),(0,n.iD)("div",V,[(0,n.Wm)(r,{showMagnitude:!1,showNumSample:!1}),(0,n.Wm)(s,{labels:l.DEMO_MODES,modelValue:l.demoMode,"onUpdate:modelValue":t[0]||(t[0]=e=>l.demoMode=e)},null,8,["labels","modelValue"]),(0,n.Wm)(c),(0,n._)("div",A,[(0,n._)("div",M,[(0,n._)("div",B,[(0,n.Wm)(u,{modelValue:l.drumType,"onUpdate:modelValue":t[1]||(t[1]=e=>l.drumType=e),options:l.DRUM_TYPES},null,8,["modelValue","options"])])])])])}const R={class:"radio-button"},O=["onClick"];function L(e,t,a,l,o,r){return(0,n.wg)(),(0,n.iD)("div",R,[((0,n.wg)(!0),(0,n.iD)(n.HY,null,(0,n.Ko)(a.options,((e,t)=>((0,n.wg)(),(0,n.iD)("div",{class:(0,i.C_)(l.optionClass(e)),key:t,onClick:()=>l.onSelect(e)},(0,i.zw)(e),11,O)))),128))])}var Z={props:["modelValue","options"],emits:["update:modelValue"],setup(e,{emit:t}){const a=t=>({"radio-button__option":!0,checked:t===e.modelValue}),l=e=>t("update:modelValue",e);return{optionClass:a,onSelect:l}}},I=a(89);const j=(0,I.Z)(Z,[["render",L]]);var W=j;const Y={class:"drum-visual"},P={id:"kick"},U=(0,n._)("path",{d:"M574.57764,389.40736H260.0344V225.27978l132.31408-6.4185q25.40964-1.23261,50.85273.00736l131.37643,6.41114Z",transform:"translate(-2.19753 -31.99738)",fill:"#4d1414"},null,-1),$={id:"drum4"},q=(0,n._)("path",{d:"M750.5062,354.32958a104.371,104.371,0,0,0-23.85384-57.43312,9.869,9.869,0,0,0-13.19853-13.22723,104.39953,104.39953,0,0,0-57.71191-24.06286,9.88179,9.88179,0,0,0-18.86719-.03282,104.39545,104.39545,0,0,0-58.04338,24.03689,9.86555,9.86555,0,0,0-13.13982,13.13982,104.36923,104.36923,0,0,0-23.98041,57.57932,9.86576,9.86576,0,0,0-.06143,18.588,104.36667,104.36667,0,0,0,24.16849,58.53351,9.86543,9.86543,0,0,0,13.081,13.05233,104.36731,104.36731,0,0,0,58.26338,23.97884,9.87464,9.87464,0,0,0,18.29221-.03042A104.36825,104.36825,0,0,0,713.386,444.4447a9.86556,9.86556,0,0,0,13.13982-13.13982,104.36734,104.36734,0,0,0,24.04184-58.38731,9.86577,9.86577,0,0,0-.06143-18.588Zm-35.34807,71.62765a9.81177,9.81177,0,0,0-7.11981,7.11981,92.37492,92.37492,0,0,1-56.14953,23.51758,9.86,9.86,0,0,0-11.20719.0268,92.36656,92.36656,0,0,1-56.25589-23.33065A9.81972,9.81972,0,0,0,577.007,425.897a92.37855,92.37855,0,0,1-23.544-57.17933,9.812,9.812,0,0,0,.04939-10.08471,92.37745,92.37745,0,0,1,23.54685-56.53509,9.81146,9.81146,0,0,0,7.11982-7.11981,92.38833,92.38833,0,0,1,57.71842-23.61579,9.79676,9.79676,0,0,0,8.782.01427,92.396,92.396,0,0,1,57.60554,23.81508,9.84678,9.84678,0,0,0,6.821,6.846A92.37977,92.37977,0,0,1,738.705,358.633a9.81211,9.81211,0,0,0,.04939,10.08471A92.3812,92.3812,0,0,1,715.15813,425.95723Z",transform:"translate(-2.19753 -31.99738)",fill:"#872222"},null,-1),z={id:"drum3"},N=(0,n._)("path",{d:"M602.105,168.39724a86.3645,86.3645,0,0,0-19.73849-47.52454A8.16634,8.16634,0,0,0,571.445,109.92749,86.38817,86.38817,0,0,0,523.68978,90.016a8.17694,8.17694,0,0,0-15.61215-.02715,86.38467,86.38467,0,0,0-48.02952,19.89,8.16351,8.16351,0,0,0-10.87289,10.87288A86.36315,86.36315,0,0,0,429.332,168.39724a8.16369,8.16369,0,0,0-.05083,15.38112A86.361,86.361,0,0,0,449.28,232.21345a8.16341,8.16341,0,0,0,10.82423,10.80049,86.36149,86.36149,0,0,0,48.21157,19.84192,8.171,8.171,0,0,0,15.13637-.02517,86.36218,86.36218,0,0,0,47.93666-19.86533,8.16351,8.16351,0,0,0,10.87289-10.87289,86.36149,86.36149,0,0,0,19.894-48.31411,8.16369,8.16369,0,0,0-.05083-15.38112Zm-29.24969,59.27018a8.119,8.119,0,0,0-5.89147,5.89148,76.43812,76.43812,0,0,1-46.4624,19.46023,8.15889,8.15889,0,0,0-9.27369.02218,76.43113,76.43113,0,0,1-46.55041-19.30556,8.12556,8.12556,0,0,0-6.13881-6.11814,76.441,76.441,0,0,1-19.48214-47.31453,8.1193,8.1193,0,0,0,.04087-8.34486,76.44012,76.44012,0,0,1,19.48446-46.78144,8.11878,8.11878,0,0,0,5.89148-5.89148,76.44914,76.44914,0,0,1,47.76062-19.5415,8.10663,8.10663,0,0,0,7.26688.0118A76.45555,76.45555,0,0,1,567.16789,119.462a8.14793,8.14793,0,0,0,5.6442,5.66488,76.44206,76.44206,0,0,1,19.52765,46.83132,8.1193,8.1193,0,0,0,.04087,8.34486A76.44321,76.44321,0,0,1,572.85528,227.66742Z",transform:"translate(-2.19753 -31.99738)",fill:"#872222"},null,-1),G={id:"drum2"},X=(0,n._)("path",{d:"M320.00167,345.14784a75.73894,75.73894,0,0,0-17.31-41.67749,7.16163,7.16163,0,0,0-9.57778-9.5986,75.75966,75.75966,0,0,0-41.8798-17.4617,7.17091,7.17091,0,0,0-13.69135-.02381,75.75659,75.75659,0,0,0-42.12034,17.44285,7.15913,7.15913,0,0,0-9.53517,9.53517,75.73766,75.73766,0,0,0-17.40187,41.78358,7.15929,7.15929,0,0,0-.04458,13.48875,75.7358,75.7358,0,0,0,17.53835,42.476,7.159,7.159,0,0,0,9.49251,9.47168,75.73627,75.73627,0,0,0,42.28,17.40073,7.16573,7.16573,0,0,0,13.27411-.02208,75.73687,75.73687,0,0,0,42.03891-17.42125,7.15914,7.15914,0,0,0,9.53517-9.53517,75.73623,75.73623,0,0,0,17.44644-42.36992,7.15929,7.15929,0,0,0-.04458-13.48875Zm-25.651,51.978a7.12008,7.12008,0,0,0-5.16663,5.16663,67.03371,67.03371,0,0,1-40.746,17.066,7.15506,7.15506,0,0,0-8.13272.01945A67.02765,67.02765,0,0,1,199.482,402.44761a7.12588,7.12588,0,0,0-5.38354-5.36541,67.03635,67.03635,0,0,1-17.08521-41.49332,7.12034,7.12034,0,0,0,.03584-7.31817,67.03553,67.03553,0,0,1,17.08724-41.02582,7.1199,7.1199,0,0,0,5.16664-5.16663A67.04341,67.04341,0,0,1,241.18755,284.941a7.10925,7.10925,0,0,0,6.37281.01035A67.049,67.049,0,0,1,289.363,302.23323a7.1455,7.1455,0,0,0,4.94978,4.96792,67.03715,67.03715,0,0,1,17.12512,41.06956,7.12031,7.12031,0,0,0,.03585,7.31817A67.03829,67.03829,0,0,1,294.35064,397.12588Z",transform:"translate(-2.19753 -31.99738)",fill:"#872222"},null,-1),K={id:"drum1"},Q=(0,n._)("path",{d:"M415.79039,172.59137a86.36452,86.36452,0,0,0-19.73848-47.52455,8.16635,8.16635,0,0,0-10.92147-10.94521,86.3882,86.3882,0,0,0-47.75523-19.91145,8.17694,8.17694,0,0,0-15.61215-.02716,86.38478,86.38478,0,0,0-48.02952,19.89,8.16351,8.16351,0,0,0-10.87289,10.87289,86.36308,86.36308,0,0,0-19.84322,47.64552,8.16368,8.16368,0,0,0-.05083,15.38111,86.36088,86.36088,0,0,0,19.99885,48.43509,8.16342,8.16342,0,0,0,10.82424,10.8005A86.36143,86.36143,0,0,0,322.00125,267.05a8.171,8.171,0,0,0,15.13638-.02518,86.3622,86.3622,0,0,0,47.93666-19.86532,8.16351,8.16351,0,0,0,10.87288-10.87289,86.36143,86.36143,0,0,0,19.89406-48.31412,8.16368,8.16368,0,0,0-.05084-15.38111Zm-29.24968,59.27018a8.119,8.119,0,0,0-5.89148,5.89147,76.43807,76.43807,0,0,1-46.46239,19.46024,8.15885,8.15885,0,0,0-9.27369.02218,76.43116,76.43116,0,0,1-46.55041-19.30556,8.12559,8.12559,0,0,0-6.13882-6.11814,76.4411,76.4411,0,0,1-19.48214-47.31453,8.11924,8.11924,0,0,0,.04087-8.34486,76.44013,76.44013,0,0,1,19.48447-46.78145,8.11871,8.11871,0,0,0,5.89147-5.89147,76.44919,76.44919,0,0,1,47.76063-19.5415,8.10661,8.10661,0,0,0,7.26687.0118,76.45547,76.45547,0,0,1,47.66722,19.70642,8.148,8.148,0,0,0,5.64421,5.66488,76.4421,76.4421,0,0,1,19.52765,46.83132,8.11924,8.11924,0,0,0,.04087,8.34486A76.44334,76.44334,0,0,1,386.54071,231.86155Z",transform:"translate(-2.19753 -31.99738)",fill:"#872222"},null,-1),J={id:"crash3"},ee=(0,n.uE)('<path d="M775.223,251.368a112.19182,112.19182,0,0,1-35.53836,31.72248q-2.51259,1.42383-5.066,2.704l-50.29193-100.524ZM628.60265,87.07059a112.18884,112.18884,0,0,0-35.59272,31.79683l91.31677,66.40307L633.80982,84.297Q631.18608,85.61009,628.60265,87.07059Z" transform="translate(-2.19753 -31.99738)" fill="#fff" opacity="0.28"></path><circle cx="681.9461" cy="153.08317" r="96.60944" fill="none" stroke="#292929" stroke-miterlimit="10"></circle><circle cx="681.9461" cy="153.08317" r="88.46644" fill="none" stroke="#292929" stroke-miterlimit="10"></circle><circle cx="681.9461" cy="153.08317" r="81.89606" fill="none" stroke="#292929" stroke-miterlimit="10"></circle><circle cx="681.9461" cy="153.08317" r="27.31341" fill="none" stroke="#292929" stroke-miterlimit="10"></circle>',5),te={id:"crash2"},ae=(0,n.uE)('<path d="M180.59254,385.54123A98.22924,98.22924,0,0,1,149.477,413.31581q-2.19988,1.24664-4.43551,2.36748l-44.033-88.01363ZM52.2193,241.69085a98.22678,98.22678,0,0,0-31.16316,27.83968l79.95228,58.13913-44.23-88.40719Q54.48121,240.4121,52.2193,241.69085Z" transform="translate(-2.19753 -31.99738)" fill="#fff" opacity="0.28"></path><circle cx="98.65061" cy="295.50598" r="84.58628" fill="none" stroke="#292929" stroke-miterlimit="10"></circle><circle cx="98.65061" cy="295.50598" r="77.45669" fill="none" stroke="#292929" stroke-miterlimit="10"></circle><circle cx="98.65061" cy="295.50598" r="71.704" fill="none" stroke="#292929" stroke-miterlimit="10"></circle><circle cx="98.65061" cy="295.50598" r="23.91423" fill="none" stroke="#292929" stroke-miterlimit="10"></circle>',5),le={id:"crash1"},ne=(0,n.uE)('<path d="M269.55236,200.137a105.40812,105.40812,0,0,1-33.38957,29.80441q-2.36065,1.33774-4.75967,2.5405l-47.25107-94.44588ZM131.7973,45.77372A105.40545,105.40545,0,0,0,98.35666,75.648l85.79539,62.38807L136.68962,43.16787Q134.22452,44.40152,131.7973,45.77372Z" transform="translate(-2.19753 -31.99738)" fill="#fff" opacity="0.28"></path><circle cx="181.78252" cy="105.86023" r="90.76804" fill="none" stroke="#292929" stroke-miterlimit="10"></circle><circle cx="181.78252" cy="105.86023" r="83.11741" fill="none" stroke="#292929" stroke-miterlimit="10"></circle><circle cx="181.78252" cy="105.86023" r="76.9443" fill="none" stroke="#292929" stroke-miterlimit="10"></circle><circle cx="181.78252" cy="105.86023" r="25.66193" fill="none" stroke="#292929" stroke-miterlimit="10"></circle>',5);function ie(e,t,a,l,o,r){return(0,n.wg)(),(0,n.iD)("div",Y,[((0,n.wg)(),(0,n.iD)("svg",{class:(0,i.C_)({[l.drumType]:!0}),"data-name":"Layer 2",xmlns:"http://www.w3.org/2000/svg",viewBox:"0 0 794.61898 442.6218"},[(0,n._)("g",P,[U,(0,n._)("ellipse",{class:(0,i.C_)(l.hitClass(6)),onClick:t[0]||(t[0]=()=>l.onDrumHit(6)),id:"highlight",cx:"415.28721",cy:"359.1808",rx:"157.0929",ry:"11.54992",fill:"#ccc"},null,2)]),(0,n._)("g",$,[(0,n._)("circle",{class:(0,i.C_)(l.hitClass(7)),onClick:t[1]||(t[1]=()=>l.onDrumHit(7)),id:"highlight-2","data-name":"highlight",cx:"643.91112",cy:"332.03022",r:"92.76431",fill:"#dcdcdc"},null,2),q]),(0,n._)("g",z,[(0,n._)("circle",{class:(0,i.C_)(l.hitClass(4)),onClick:t[2]||(t[2]=()=>l.onDrumHit(4)),id:"highlight-3","data-name":"highlight",cx:"513.52095",cy:"144.42474",r:"76.76026",fill:"#dcdcdc"},null,2),N]),(0,n._)("g",G,[(0,n._)("circle",{class:(0,i.C_)(l.hitClass(3)),onClick:t[3]||(t[3]=()=>l.onDrumHit(3)),id:"highlight-4","data-name":"highlight",cx:"242.04598",cy:"320.18803",r:"67.31627",fill:"#dcdcdc"},null,2),X]),(0,n._)("g",K,[(0,n._)("circle",{class:(0,i.C_)(l.hitClass(1)),onClick:t[4]||(t[4]=()=>l.onDrumHit(1)),id:"highlight-5","data-name":"highlight",cx:"327.20638",cy:"148.61887",r:"76.76026",fill:"#dcdcdc"},null,2),Q]),(0,n._)("g",J,[(0,n._)("circle",{class:(0,i.C_)(l.hitClass(5)),onClick:t[5]||(t[5]=()=>l.onDrumHit(5)),id:"highlight-6","data-name":"highlight",cx:"681.9461",cy:"153.08317",r:"112.65323",fill:"#c3a22e"},null,2),ee]),(0,n._)("g",te,[(0,n._)("circle",{class:(0,i.C_)(l.hitClass(2)),onClick:t[6]||(t[6]=()=>l.onDrumHit(2)),id:"highlight-7","data-name":"highlight",cx:"98.65061",cy:"295.50598",r:"98.6334",fill:"#c3a22e"},null,2),ae]),(0,n._)("g",le,[(0,n._)("circle",{class:(0,i.C_)(l.hitClass(0)),onClick:t[7]||(t[7]=()=>l.onDrumHit(0)),id:"highlight-8","data-name":"highlight",cx:"181.78252",cy:"105.86023",r:"105.84177",fill:"#c3a22e"},null,2),ne])],2))])}var oe={setup(){const e=e=>({hit:m[e]});return{hitClass:e,drumType:h,onDrumHit:k,highlights:m}}};const re=(0,I.Z)(oe,[["render",ie]]);var se=re;const ce={class:"settings"},ue={key:0,class:"settings__item"},de=["min","max"],pe={key:1,class:"settings__item"},ve=["min","max"],ge={key:2,class:"settings__item"},me=["min","max"],he={key:3,class:"settings__item"},we=["min","max"],fe={key:4,class:"magnitude"},_e={class:"bar"};function ke(e,t,a,o,r,s){return(0,n.wg)(),(0,n.iD)("div",ce,[a.showNumSample?((0,n.wg)(),(0,n.iD)("div",ue,[(0,n._)("span",null,(0,i.zw)(o.numSample)+" sample per capture",1),(0,n.wy)((0,n._)("input",{type:"range",min:o.SAMPLE_RAMGE[0],max:o.SAMPLE_RAMGE[1],"onUpdate:modelValue":t[0]||(t[0]=e=>o.numSample=e),onMouseup:t[1]||(t[1]=()=>o.updateDeviceParam())},null,40,de),[[l.nr,o.numSample]])])):(0,n.kq)("",!0),a.showTestResponseTime?((0,n.wg)(),(0,n.iD)("div",pe,[(0,n._)("span",null,"Test response time: "+(0,i.zw)(o.testResponseTime),1),(0,n.wy)((0,n._)("input",{type:"range",min:o.TEST_RESPONSE_TIME_RANGE[0],max:o.TEST_RESPONSE_TIME_RANGE[1],step:"1.0","onUpdate:modelValue":t[2]||(t[2]=e=>o.testResponseTime=e),onMouseup:t[3]||(t[3]=()=>o.updateDeviceParam())},null,40,ve),[[l.nr,o.testResponseTime]])])):(0,n.kq)("",!0),a.showCooldown?((0,n.wg)(),(0,n.iD)("div",ge,[(0,n._)("span",null,"Cooldown: "+(0,i.zw)(o.cooldown),1),(0,n.wy)((0,n._)("input",{type:"range",min:o.COOLDOWN_RANGE[0],max:o.COOLDOWN_RANGE[1],step:"1.0","onUpdate:modelValue":t[4]||(t[4]=e=>o.cooldown=e),onMouseup:t[5]||(t[5]=()=>o.updateDeviceParam())},null,40,me),[[l.nr,o.cooldown]])])):(0,n.kq)("",!0),a.showThreshold?((0,n.wg)(),(0,n.iD)("div",he,[(0,n._)("span",null,"Threshold: "+(0,i.zw)(o.threshold),1),(0,n.wy)((0,n._)("input",{type:"range",min:o.THRESHOLD_RANGE[0],max:o.THRESHOLD_RANGE[1],step:"0.01","onUpdate:modelValue":t[6]||(t[6]=e=>o.threshold=e),onMouseup:t[7]||(t[7]=()=>o.updateDeviceParam())},null,40,we),[[l.nr,o.threshold]])])):(0,n.kq)("",!0),a.showMagnitude?((0,n.wg)(),(0,n.iD)("div",fe,[(0,n._)("span",null,"Magnitude: "+(0,i.zw)(o.magnitude),1),(0,n._)("div",_e,[(0,n._)("span",{style:(0,i.j5)(o.getMagnitudeBarStyle())},null,4),(0,n._)("span",{class:"mark",style:(0,i.j5)({left:`calc(${o.threshold/o.THRESHOLD_RANGE[1]} * 100%)`})},null,4)])])):(0,n.kq)("",!0)])}a(7658);const ye=[{index:0,label:"Demo"},{index:1,label:"Data Collection"}],Ce=(0,v.qj)({index:0,label:ye[0]}),De=e=>{Ce.index=e.index,Ce.label=e.label,Ze()},be="f30c5d5f-ec5a-4c1d-94c5-46e35d810dc5",xe="2f925c9d-0a5b-4217-8e63-2d527c9211c1",Se="f8edf338-6bbd-4c3b-bf16-d8d2b6cdaa6e",Te={left:(0,v.qj)({name:"",gatt:null,web2boardChar:null,readyToWrite:!0}),right:(0,v.qj)({name:"",gatt:null,web2boardChar:null,readyToWrite:!0})},Fe=(0,v.iH)(40),Ee=[20,200],Ve=(0,v.iH)(200),Ae=[100,900],Me=new TextEncoder("utf-8"),Be=new TextDecoder("utf-8"),He=async e=>{let t=null,a=null;try{let l=await navigator.bluetooth.requestDevice({filters:[{services:[be]}]});const n="DRUM_L"==l.name?"left":"right";l.addEventListener("gattserverdisconnected",(()=>{console.log(l.name," disconnect"),Te[n].gatt=null,Te[n].web2boardChar=null}));let i=await l.gatt.connect(),o=await i.getPrimaryService(be);t=await o.getCharacteristic(xe),t.properties.notify&&await t.startNotifications(),a=await o.getCharacteristic(Se),a.properties.notify&&(a.addEventListener("characteristicvaluechanged",Oe),await a.startNotifications()),Te[n].name=l.name,Te[n].gatt=l.gatt,Te[n].web2boardChar=t,Te[n].readyToWrite=!0,e&&e(!0),console.log(l.name,"connected."),Ze()}catch(l){e&&e(!1),console.warn("Error occurs during BT device connection.",l)}},Re=async e=>{e.gatt.connected?await e.gatt.disconnect():console.log("device is already disconnected")},Oe=e=>{const t=Be.decode(e.target.value);"m0"===t.slice(1,3)?k(parseInt(t[3])):(Le().length<2||"0"===t[0])&&Je(t.slice(1))},Le=()=>Object.keys(Te).filter((e=>Te[e].gatt&&Te[e].gatt.connected)).map((e=>Te[e])),Ze=(e=Ce.index,t=Ye.value,a=ft.value,l=Pe.value,n=Ve.value,i=We.value,o=Fe.value)=>{let r=0;const s=a===wt[0]?0:1,c=t?1:0,u=(o.toString().length<3?"0":"")+o.toString(),d=(l.toString().length<2?"0":"")+l.toString(),p=`${e}${s}${c}${d}${u}${n}${parseFloat(i).toFixed(2)}`.substring(0,20),v=Le();return console.log(p),v.forEach((async t=>{try{if(!t.readyToWrite)return void console.warn("previous write operation still in progress, please update device params again later.");t.readyToWrite=!1,await t.web2boardChar.writeValueWithResponse(Me.encode(p)),t.readyToWrite=!0,console.log(`[-> ${t.name}]: view(${e}) demoMode(${s}) capture(${c}) data(${d}) cooldown(${u}) test response time(${n}) threshold(${i})`),r++}catch(a){console.warn("Error on updating device param. ",a)}})),r==v.length},Ie=[5,50],je=[.1,.5],We=(0,v.iH)(.16);let Ye=(0,v.iH)(!1);const Pe=(0,v.iH)(15),Ue=(0,v.qj)({aX:0,aY:0,aZ:0,gX:0,gY:0,gZ:0}),$e=(0,n.Fl)((()=>((Ue.aX+Ue.aY+Ue.aZ+Ue.gX+Ue.gY+Ue.gZ)/6).toFixed(3))),qe=(0,v.iH)([]),ze=(0,v.iH)({}),Ne=(0,v.iH)(-1),Ge=(0,v.qj)({t:0,val:[]}),Xe=["Main","Extracts"],Ke=(0,v.iH)(Xe[0]);let Qe=[];const Je=async e=>{if("0"===e[3]){const[t,a,l,n,i,o]=e.slice(4).split(",").map((e=>parseFloat(e)));Ue.aX=Math.abs(t/4),Ue.aY=Math.abs(a/4),Ue.aZ=Math.abs(l/4),Ue.gX=Math.abs(n/2e3),Ue.gY=Math.abs(i/2e3),Ue.gZ=Math.abs(o/2e3),Ge.val.length>=Pe.value&&Ge.val.splice(0,Ge.val.length-Pe.value+1),Ge.t=Date.now(),Ge.val.push(at(e))}else if("1"===e[3])Qe.push(e.slice(4).split(",").map((e=>parseFloat(e))));else if("2"===e[3]){const e=Date.now();qe.value.push({t:e,val:Qe}),Ke.value===Xe[1]&&(ze.value[e]=!0),Ge.val.push(...Qe),console.log(qe.value),Qe=[]}},et=e=>((e+4)/8).toFixed(3),tt=e=>((e+2e3)/4e3).toFixed(3),at=e=>{const[t,a,l,n,i,o]=e.split(",").map((e=>parseFloat(e)));return[et(t),et(a),et(l),tt(n),tt(i),tt(o)]},lt=()=>{Le().length&&(Ye.value=!Ye.value,Ze(),Ye.value&&(Ne.value=-1))},nt=()=>{Ye.value=!1},it=()=>{Ke.value===Xe[0]?(qe.value=qe.value.filter((e=>ze.value[e.t])),ze.value[Ne.value]||(Ne.value=-1)):(qe.value=qe.value.filter((e=>!ze.value[e.t])),ze.value[Ne.value]&&(Ne.value=-1),ze.value={})},ot=(e,t)=>{const a=qe.value.findIndex((t=>t.t===e));qe.value.splice(a,1),Ne.value===e&&(Ne.value=-1),t&&t.stopPropagation()},rt=()=>Le().length;var st={props:{showThreshold:{type:Boolean,default:!0},showMagnitude:{type:Boolean,default:!0},showTestResponseTime:{type:Boolean,default:!0},showCooldown:{type:Boolean,default:!0},showNumSample:{type:Boolean,default:!0}},setup(){const e=()=>({width:`calc(${Math.min(1,$e.value/je[1])} * 100%)`,backgroundColor:$e.value>We.value?"#42ff75":"red"});return{getMagnitudeBarStyle:e,threshold:We,THRESHOLD_RANGE:je,cooldown:Fe,COOLDOWN_RANGE:Ee,numSample:Pe,SAMPLE_RAMGE:Ie,magnitude:$e,testResponseTime:Ve,TEST_RESPONSE_TIME_RANGE:Ae,updateDeviceParam:Ze}}};const ct=(0,I.Z)(st,[["render",ke]]);var ut=ct;const dt={class:"label before"},pt={class:"label after"};function vt(e,t,a,l,o,r){return(0,n.wg)(),(0,n.iD)("div",{class:(0,i.C_)(["toggle-button",{checked:a.modelValue===a.labels[1]}])},[(0,n._)("span",dt,(0,i.zw)(a.labels[0]),1),(0,n._)("span",{class:"toggle",onClick:t[0]||(t[0]=(...e)=>l.onToggle&&l.onToggle(...e))}),(0,n._)("span",pt,(0,i.zw)(a.labels[1]),1)],2)}var gt={props:{labels:Array,modelValue:String},emits:["update:modelValue"],setup(e,{emit:t}){const a=()=>{t("update:modelValue",e.modelValue===e.labels[0]?e.labels[1]:e.labels[0]),Ze()};return{onToggle:a}}};const mt=(0,I.Z)(gt,[["render",vt]]);var ht=mt;const wt=["Test response","Inference mode"],ft=(0,v.iH)(wt[1]);var _t={components:{DrumSettings:ut,DrumVisual:se,RadioButton:W,ToggleButton:ht},setup(){return{drumType:h,DRUM_TYPES:g,demoMode:ft,DEMO_MODES:wt}}};const kt=(0,I.Z)(_t,[["render",H]]);var yt=kt;const Ct={class:"data-collection-view"},Dt={class:"captures"},bt={class:"captures__header"},xt={class:"captures__records"},St={class:"captures__records__wrapper"},Tt={key:0,class:"capture__records__list"},Ft=["onClick"],Et={class:"time"},Vt=["onClick"],At=["onClick"],Mt={key:1,class:"capture__records__list"},Bt=["onClick"],Ht={class:"time"},Rt=["onClick"],Ot=["onClick"],Lt={class:"capture-visual"};function Zt(e,t,a,l,o,r){const s=(0,n.up)("DrumSettings"),c=(0,n.up)("GenericButton"),u=(0,n.up)("RadioButton"),d=(0,n.up)("DataVisual");return(0,n.wg)(),(0,n.iD)("div",Ct,[(0,n.Wm)(s,{showCooldown:!1,showTestResponseTime:!1}),(0,n._)("div",Dt,[(0,n._)("div",bt,[(0,n._)("h4",null,"Captured: "+(0,i.zw)(l.captureListView===l.LIST_VIEWS[0]?l.mainCapBF.length:l.filteredCapBF.length),1),(0,n.Wm)(c,{class:(0,i.C_)({disabled:!l.hasAvailableDevices()}),onClick:l.startCapture},{default:(0,n.w5)((()=>[(0,n.Uk)((0,i.zw)(l.captureStarted?"Pause":"Start"),1)])),_:1},8,["class","onClick"]),(0,n.Wm)(c,{class:(0,i.C_)({disabled:!l.capturedBuffer.length}),onClick:l.resetCapture},{default:(0,n.w5)((()=>[(0,n.Uk)("Clear")])),_:1},8,["class","onClick"]),l.captureListView===l.LIST_VIEWS[0]?((0,n.wg)(),(0,n.iD)("a",{key:0,class:(0,i.C_)(["downloadCSV",{disabled:!l.mainCapBF.length}]),download:"capture.csv",ref:"download",onClick:t[0]||(t[0]=(...e)=>l.saveCapture&&l.saveCapture(...e))},"Save gesture as .csv",2)):((0,n.wg)(),(0,n.iD)("a",{key:1,class:(0,i.C_)(["updateUnknownCSV",{disabled:!l.filteredCapBF.length}]),download:"unknown.csv",ref:"downloadFilterCapture",onClick:t[1]||(t[1]=(...e)=>l.saveFilteredCapture&&l.saveFilteredCapture(...e))},"Save extracted gesture as .csv",2))]),(0,n._)("div",xt,[(0,n._)("div",St,[(0,n.Wm)(u,{class:"capture__records__view-toggle",modelValue:l.captureListView,"onUpdate:modelValue":t[2]||(t[2]=e=>l.captureListView=e),options:l.LIST_VIEWS},null,8,["modelValue","options"]),l.captureListView===l.LIST_VIEWS[0]?((0,n.wg)(),(0,n.iD)("div",Tt,[((0,n.wg)(!0),(0,n.iD)(n.HY,null,(0,n.Ko)(l.mainCapBF,(({t:e},t)=>((0,n.wg)(),(0,n.iD)("div",{class:(0,i.C_)(l.captureItemClass(e)),key:t,onClick:()=>l.selectCapturedBuffer(e)},[(0,n.Uk)(" captured at: "),(0,n._)("span",Et,(0,i.zw)(e),1),(0,n._)("span",{class:"delete",onClick:t=>l.removeCapturedItem(e,t)},"✕",8,Vt),(0,n._)("span",{class:"filter-capture",onClick:t=>l.setUnknownCapture(e,t)},"→",8,At)],10,Ft)))),128)),(0,n._)("div",{class:"new-capture-indicator",style:(0,i.j5)(l.getIndicatorStyle())},null,4)])):((0,n.wg)(),(0,n.iD)("div",Mt,[((0,n.wg)(!0),(0,n.iD)(n.HY,null,(0,n.Ko)(l.filteredCapBF,(({t:e},t)=>((0,n.wg)(),(0,n.iD)("div",{class:(0,i.C_)(l.captureItemClass(e)),key:t,onClick:()=>l.selectCapturedBuffer(e)},[(0,n.Uk)(" captured at: "),(0,n._)("span",Ht,(0,i.zw)(e),1),(0,n._)("span",{class:"delete",onClick:t=>l.removeCapturedItem(e,t)},"✕",8,Rt),(0,n._)("span",{class:"filter-capture",onClick:t=>l.setUnknownCapture(e,t)},"←",8,Ot)],10,Bt)))),128)),(0,n._)("div",{class:"new-capture-indicator",style:(0,i.j5)(l.getIndicatorStyle())},null,4)]))]),(0,n._)("div",Lt,[(0,n._)("h4",null,[(0,n.Uk)("View: "),(0,n._)("span",null,(0,i.zw)(l.getCurrentView()),1)]),(0,n.Wm)(d,{sensorData:l.getSensorData()},null,8,["sensorData"])])])])])}const It={class:"data-visual"},jt={ref:"root",width:"100%",height:"100%"},Wt={key:0},Yt=["d","stroke"],Pt={class:"tick-x"},Ut={class:"legend"};function $t(e,t,a,l,o,r){return(0,n.wg)(),(0,n.iD)("div",It,[((0,n.wg)(),(0,n.iD)("svg",jt,[a.sensorData&&a.sensorData.val[0]&&1===a.sensorData.val[0].length?((0,n.wg)(),(0,n.iD)("text",Wt,"test")):((0,n.wg)(!0),(0,n.iD)(n.HY,{key:1},(0,n.Ko)(l.paths,(({color:e,d:t},a)=>((0,n.wg)(),(0,n.iD)("path",{key:a,d:t,fill:"none",stroke:e,"stroke-width":"1"},null,8,Yt)))),128)),(0,n._)("line",(0,i.vs)((0,n.F4)(l.middleLineStyle())),null,16),(0,n._)("g",Pt,[((0,n.wg)(!0),(0,n.iD)(n.HY,null,(0,n.Ko)(parseInt(l.numSample),((e,t)=>((0,n.wg)(),(0,n.iD)("line",(0,n.dG)({key:t},l.tickXAttr(e-1)),null,16)))),128)),((0,n.wg)(!0),(0,n.iD)(n.HY,null,(0,n.Ko)(l.tickX,((e,t)=>((0,n.wg)(),(0,n.iD)("text",(0,n.dG)({key:t},l.tickXTextAttr(e-1)),(0,i.zw)(e),17)))),128))]),((0,n.wg)(!0),(0,n.iD)(n.HY,null,(0,n.Ko)(l.tickY,((e,t)=>((0,n.wg)(),(0,n.iD)("g",{class:"tick-y",key:t},[(0,n._)("line",(0,i.vs)((0,n.F4)(l.tickYAttr(t))),null,16),(0,n._)("text",(0,i.vs)((0,n.F4)(l.tickYTextAttr(t))),(0,i.zw)(e),17)])))),128))],512)),(0,n._)("div",Ut,[((0,n.wg)(),(0,n.iD)(n.HY,null,(0,n.Ko)(["aX","aY","aZ","gX","gY","gZ"],((e,t)=>(0,n._)("p",{key:t},[(0,n._)("span",{style:(0,i.j5)({backgroundColor:l.colorScheme[t]})},null,4),(0,n.Uk)(" "+(0,i.zw)(e),1)]))),64))])])}const qt=Object.assign({},a(8648),a(6796)),zt={top:50,right:15,bottom:40,left:40},Nt=["#0352fc","#3d7bff","#7da7ff","#ff5e00","#ff8842","#ffaf80","#9602d1","#c902e3","#fc03f4"],Gt=(e,t,a,l=0)=>Array.from({length:Math.ceil((t-e)/a)+1},((t,n)=>(e+n*a).toFixed(l)));var Xt={props:{sensorData:Object,default:null},setup(e){const t=(0,v.iH)(null),a=(0,v.qj)({width:0,height:0}),l=(0,n.Fl)((()=>{const e=12;let t=Gt(1,Pe.value,Math.ceil(Pe.value/e));if(Pe.value>e){t[t.length-1]>Pe.value&&t.pop();const e=Pe.value-t[t.length-1];e&&e<2?t.splice(t.length-1,1,Pe.value):t.push(Pe.value)}return t})),i=Gt(0,1,.1,1),o=(0,n.Fl)((()=>{if(!e.sensorData)return[];const t=qt.scaleLinear().domain([0,Pe.value-1]).range([zt.left,a.width-zt.right]),l=qt.scaleLinear().domain([i[0],i[i.length-1]]).range([a.height-zt.bottom,zt.top]),n=qt.line().x((e=>t(e.x))).y((e=>l(e.y))),o=[[],[],[],[],[],[]];return e.sensorData.val.forEach(((e,t)=>{e.forEach(((e,a)=>{o[a].push({x:t,y:e})}))})),o.filter((e=>n(e)&&-1===n(e).indexOf("NaN"))).map(((e,t)=>({color:Nt[t],d:n(e)})))})),r=()=>{t.value&&(a.width=parseInt(t.value.getBoundingClientRect().width),a.height=parseInt(t.value.getBoundingClientRect().height))},s=e=>{const t=a.width-zt.right-zt.left,n=l.value.includes((e+1).toString());return{stroke:"#FFFFFF",opacity:n?.3:.1,x1:zt.left+e/(Pe.value-1)*t,y1:zt.top,x2:zt.left+e/(Pe.value-1)*t,y2:a.height-24}},c=e=>{const t=a.width-zt.right-zt.left;return{fill:"#FFFFFF","font-size":12,"text-anchor":"middle",transform:`translate(${zt.left+e/(Pe.value-1)*t}, ${a.height-6})`}},u=e=>{const t=a.height-zt.top-zt.bottom;return{stroke:"#FFFFFF",opacity:.1,x1:zt.left-5,y1:zt.top+t-e/(i.length-1)*t,x2:a.width-zt.right,y2:zt.top+t-e/(i.length-1)*t}},d=e=>{const t=a.height-zt.top-zt.bottom;return{fill:"#FFFFFF","font-size":12,"text-anchor":"end",transform:`translate(${zt.left-10}, ${zt.top+t-e/(i.length-1)*t+5})`}},p=()=>{const e=zt.top+(a.height-zt.top-zt.bottom)/2;return{stroke:"#FFFFFF",strokeWidth:1.5,x1:zt.left,y1:e,x2:a.width-zt.right,y2:e}},g=new ResizeObserver(r);return(0,n.bv)((()=>{g.observe(t.value),r()})),{root:t,paths:o,box:a,tickX:l,tickXAttr:s,tickXTextAttr:c,tickY:i,tickYAttr:u,tickYTextAttr:d,middleLineStyle:p,numSample:Pe,colorScheme:Nt}}};const Kt=(0,I.Z)(Xt,[["render",$t]]);var Qt=Kt;const Jt=["disabled"];function ea(e,t,a,l,i,o){return(0,n.wg)(),(0,n.iD)("div",{class:"generic-button",disabled:a.disabled,onClick:t[0]||(t[0]=(...e)=>l.onClick&&l.onClick(...e))},[(0,n.WI)(e.$slots,"default")],8,Jt)}var ta={props:{disabled:{type:Boolean,default:!1}},emits:["click"],setup(e,{emit:t}){const a=e=>t("click",e);return{onClick:a}}};const aa=(0,I.Z)(ta,[["render",ea]]);var la=aa,na={components:{DrumSettings:ut,GenericButton:la,RadioButton:W,DataVisual:Qt},setup(){const e=(0,n.Fl)((()=>qe.value.filter((e=>!ze.value[e.t])))),t=(0,n.Fl)((()=>qe.value.filter((e=>ze.value[e.t])))),a=e=>({captures__records__item:!0,selected:e===Ne.value,disabled:Ye.value,isUnknown:ze.value[e]}),l=(e,t)=>{ze.value[e]=!ze.value[e],t.stopPropagation()},i=()=>{const e=Ye.value?Math.min(1,$e.value/We.value):0;return{width:`calc(${e} * 100%)`}},o=e=>{Ne.value=Ne.value===e?-1:e},r=()=>Ne.value>=0?qe.value.filter((e=>e.t===Ne.value))[0]:rt()?Ge:null,s=()=>Ne.value>=0?"captured at "+Ne.value:rt()?"live":"",c=(0,v.iH)(null),u=()=>{nt();const t="aX,aY,aZ,gX,gY,gZ\r\n"+e.value.map((e=>e.val.join("\r\n"))).join("\r\n\n"),a=new Blob([t],{type:"text/csv;charset=utf-8;"});c.value.href=URL.createObjectURL(a)},d=(0,v.iH)(null),p=()=>{const e="aX,aY,aZ,gX,gY,gZ\r\n"+t.value.map((e=>e.val.join("\r\n"))).join("\r\n\n"),a=new Blob([e],{type:"text/csv;charset=utf-8;"});d.value.href=URL.createObjectURL(a)};return(0,n.Jd)((()=>{Ye.value=!1})),{mainCapBF:e,filteredCapBF:t,capturedBuffer:qe,startCapture:lt,resetCapture:it,captureStarted:Ye,selectedCapTime:Ne,removeCapturedItem:ot,hasAvailableDevices:rt,download:c,downloadFilterCapture:d,captureItemClass:a,setUnknownCapture:l,getIndicatorStyle:i,selectCapturedBuffer:o,saveCapture:u,saveFilteredCapture:p,getSensorData:r,getCurrentView:s,threshold:We,magnitude:$e,captureListView:Ke,LIST_VIEWS:Xe,filterCaptureTRef:ze}}};const ia=(0,I.Z)(na,[["render",Zt]]);var oa=ia;const ra={class:"device-panel"},sa={class:"device-panel__content"},ca={class:"device-panel__entry"},ua=(0,n._)("svg",{version:"1.1",id:"Layer_1",xmlns:"http://www.w3.org/2000/svg","xmlns:xlink":"http://www.w3.org/1999/xlink",x:"0px",y:"0px",viewBox:"0 0 20 20","enable-background":"new 0 0 20 20","xml:space":"preserve"},[(0,n._)("polyline",{fill:"none",stroke:"#D1D3D4","stroke-width":"1.5","stroke-miterlimit":"10",points:"6.35,6.04 13.65,13.34 10.14,16.85 \r\n          10.14,3.15 13.63,6.65 6.67,13.62 "})],-1);function da(e,t,a,l,o,r){const s=(0,n.up)("GenericButton");return(0,n.wg)(),(0,n.iD)("div",ra,[(0,n._)("div",{class:(0,i.C_)(["device-panel-container",{inView:l.inView}]),onMouseover:t[0]||(t[0]=()=>l.inView=!0),onMouseout:t[1]||(t[1]=()=>l.inView=!1)},[(0,n._)("div",sa,[((0,n.wg)(!0),(0,n.iD)(n.HY,null,(0,n.Ko)(l.connectedDevices,((e,t)=>((0,n.wg)(),(0,n.j4)(s,{class:"device-panel__button connected",key:t,onClick:()=>l.disconnectBTDevice(e)},{default:(0,n.w5)((()=>[(0,n.Uk)((0,i.zw)(e.name),1)])),_:2},1032,["onClick"])))),128)),l.connectedDevices.length<2?((0,n.wg)(),(0,n.j4)(s,{key:0,class:"device-panel__button",style:(0,i.j5)({pointerEvents:l.connectInProgress?"none":"all",opacity:l.connectInProgress?.5:1}),onClick:l.clickConnectButton},{default:(0,n.w5)((()=>[(0,n.Uk)((0,i.zw)(l.connectInProgress?"connecting...":"+ Add peripheral"),1)])),_:1},8,["style","onClick"])):(0,n.kq)("",!0)]),(0,n._)("div",ca,[ua,(0,n._)("span",null,(0,i.zw)(l.connectedDevices.length),1)])],34)])}var pa={components:{GenericButton:la},setup(){const e=(0,v.iH)(!1),t=(0,v.iH)(!1),a=(0,n.Fl)((()=>Le())),l=()=>{t.value=!0,He((()=>{t.value=!1,e.value=!1}))};return{connectInProgress:t,clickConnectButton:l,disconnectBTDevice:Re,connectedDevices:a,inView:e}}};const va=(0,I.Z)(pa,[["render",da]]);var ga=va,ma={name:"drum-web-app",components:{DemoView:yt,DataCollectView:oa,DevicePanel:ga},setup(){navigator.bluetooth||console.warn("Please use a browser that supports web bluetooth api");const e=e=>({navbar__option:!0,current:e.index===Ce.index});return{view:Ce,VIEWS:ye,setView:De,navbarOptionClass:e,loadAudioDatas:F,audioReady:C,isAudioLoading:D}}};const ha=(0,I.Z)(ma,[["render",p]]);var wa=ha;(0,l.ri)(wa).mount("#app")}},t={};function a(l){var n=t[l];if(void 0!==n)return n.exports;var i=t[l]={exports:{}};return e[l](i,i.exports,a),i.exports}a.m=e,function(){var e=[];a.O=function(t,l,n,i){if(!l){var o=1/0;for(u=0;u<e.length;u++){l=e[u][0],n=e[u][1],i=e[u][2];for(var r=!0,s=0;s<l.length;s++)(!1&i||o>=i)&&Object.keys(a.O).every((function(e){return a.O[e](l[s])}))?l.splice(s--,1):(r=!1,i<o&&(o=i));if(r){e.splice(u--,1);var c=n();void 0!==c&&(t=c)}}return t}i=i||0;for(var u=e.length;u>0&&e[u-1][2]>i;u--)e[u]=e[u-1];e[u]=[l,n,i]}}(),function(){a.n=function(e){var t=e&&e.__esModule?function(){return e["default"]}:function(){return e};return a.d(t,{a:t}),t}}(),function(){a.d=function(e,t){for(var l in t)a.o(t,l)&&!a.o(e,l)&&Object.defineProperty(e,l,{enumerable:!0,get:t[l]})}}(),function(){a.g=function(){if("object"===typeof globalThis)return globalThis;try{return this||new Function("return this")()}catch(e){if("object"===typeof window)return window}}()}(),function(){a.o=function(e,t){return Object.prototype.hasOwnProperty.call(e,t)}}(),function(){a.r=function(e){"undefined"!==typeof Symbol&&Symbol.toStringTag&&Object.defineProperty(e,Symbol.toStringTag,{value:"Module"}),Object.defineProperty(e,"__esModule",{value:!0})}}(),function(){var e={143:0};a.O.j=function(t){return 0===e[t]};var t=function(t,l){var n,i,o=l[0],r=l[1],s=l[2],c=0;if(o.some((function(t){return 0!==e[t]}))){for(n in r)a.o(r,n)&&(a.m[n]=r[n]);if(s)var u=s(a)}for(t&&t(l);c<o.length;c++)i=o[c],a.o(e,i)&&e[i]&&e[i][0](),e[i]=0;return a.O(u)},l=self["webpackChunkdrum_webapp"]=self["webpackChunkdrum_webapp"]||[];l.forEach(t.bind(null,0)),l.push=t.bind(null,l.push.bind(l))}();var l=a.O(void 0,[998],(function(){return a(84)}));l=a.O(l)})();
//# sourceMappingURL=app.4905d6e9.js.map