var implementation=0

var implementation_2=0

document.querySelector(".col-6 .btn.btn-css-1").addEventListener("click",function(){
  document.querySelectorAll(".row .col-6")[0].style.display="None"
  document.querySelectorAll(".row .col-6")[1].style.display="None"
  document.querySelector(".main .row.row_css").style.height="650px"
  document.querySelector(".implement").style.display="block"
  document.querySelector(".q_mix_base").style.position="relative"
  document.querySelector(".q_mix_base").style.top="-750px"
  if (implementation_2===1){
    document.querySelector('.implement_q_mix').style.top="-1500px"
  }
  implementation=1

})

var get_condition="None"

var counter_btn_1=0

var counter_btn_2=0

var set="None"

var active="None"

document.querySelector(".btn.btn-net-1").addEventListener("click",function(event){
  if (get_condition==="None"){
    if (active){
      document.querySelector('#c_net_1').style.display="None"
      document.querySelector('#c_net_1_2').style.display="None"
      document.querySelector('#c_net_2').style.display="None"
      document.querySelector('#c_net_2_2').style.display="None"
      document.querySelector('#c_net_3').style.display="None"
      document.querySelector('#c_net_3_2').style.display="None"
      document.querySelector('.start_crossover').style.display="None"
      document.querySelector('.text_cross_net_1').style.display="None"
      document.querySelector('.text_cross_net_2').style.display="None"
      document.querySelector('.text_cross_net_3').style.display="None"
      document.querySelector('.text_cross_net_1_2').style.display="None"
      document.querySelector('.text_cross_net_2_2').style.display="None"
    }
    document.querySelector(".container-col-8-1").style.display="block"
    document.querySelector('.network-1-text').style.backgroundColor="chartreuse"
    document.querySelector('.btn.add-connect-btn').style.display="inline-block"
    document.querySelector('.btn.add-node-btn').style.display="inline-block"
    document.querySelector('.btn.crossover-btn').style.display="inline-block"
    get_condition = "1"
    set=1
  }
})
document.querySelector(".btn.btn-net-2").addEventListener("click",function(event){
  if (get_condition==="None"){
    if (active){
      document.querySelector('#c_net_1').style.display="None"
      document.querySelector('#c_net_1_2').style.display="None"
      document.querySelector('#c_net_2').style.display="None"
      document.querySelector('#c_net_2_2').style.display="None"
      document.querySelector('#c_net_3').style.display="None"
      document.querySelector('#c_net_3_2').style.display="None"
      document.querySelector('.start_crossover').style.display="None"
      document.querySelector('.text_cross_net_1').style.display="None"
      document.querySelector('.text_cross_net_2').style.display="None"
      document.querySelector('.text_cross_net_3').style.display="None"
      document.querySelector('.text_cross_net_1_2').style.display="None"
      document.querySelector('.text_cross_net_2_2').style.display="None"
    }
    document.querySelector(".container-col-8-2").style.display="block"
    document.querySelector('.network-2-text').style.backgroundColor="chartreuse"
    document.querySelector('.btn.add-connect-btn').style.display="inline-block"
    document.querySelector('.btn.add-node-btn').style.display="inline-block"
    document.querySelector('.btn.crossover-btn').style.display="inline-block"
    get_condition = "2"
    set=2
  }
})


document.querySelector(".unset-net").addEventListener("click",function(event){
  if (get_condition==="1"){
    document.querySelector(".container-col-8-1").style.display="None"
    document.querySelector('.btn.add-connect-btn').style.display="None"
    document.querySelector('.btn.add-node-btn').style.display="None"
    document.querySelector('.btn.crossover-btn').style.display="None"
  }
  else if (get_condition==="2"){
    document.querySelector(".container-col-8-2").style.display="None"
    document.querySelector('.btn.add-connect-btn').style.display="None"
    document.querySelector('.btn.add-node-btn').style.display="None"
    document.querySelector('.btn.crossover-btn').style.display="None"
  }
  else{
    document.querySelector(".container-col-8-1").style.display="None"
    document.querySelector(".container-col-8-2").style.display="None"
    document.querySelector('.btn.add-connect-btn').style.display="None"
    document.querySelector('.btn.add-node-btn').style.display="None"
    document.querySelector('.btn.crossover-btn').style.display="None"
  }
  if (active){
    document.querySelector('#c_net_1').style.display="None"
    document.querySelector('#c_net_1_2').style.display="None"
    document.querySelector('#c_net_2').style.display="None"
    document.querySelector('#c_net_2_2').style.display="None"
    document.querySelector('#c_net_3').style.display="None"
    document.querySelector('#c_net_3_2').style.display="None"
    document.querySelector('.start_crossover').style.display="None"
    document.querySelector('.text_cross_net_1').style.display="None"
    document.querySelector('.text_cross_net_2').style.display="None"
    document.querySelector('.text_cross_net_3').style.display="None"
    document.querySelector('.text_cross_net_1_2').style.display="None"
    document.querySelector('.text_cross_net_2_2').style.display="None"
  }
  get_condition="None"
  set="None"
})


document.querySelector(".keep-btn").addEventListener("click",function(event){
  if (active){
    document.querySelector('#c_net_1').style.display="None"
    document.querySelector('#c_net_1_2').style.display="None"
    document.querySelector('#c_net_2').style.display="None"
    document.querySelector('#c_net_2_2').style.display="None"
    document.querySelector('#c_net_3').style.display="None"
    document.querySelector('#c_net_3_2').style.display="None"
    document.querySelector('.start_crossover').style.display="None"
    document.querySelector('.text_cross_net_1').style.display="None"
    document.querySelector('.text_cross_net_2').style.display="None"
    document.querySelector('.text_cross_net_3').style.display="None"
    document.querySelector('.text_cross_net_1_2').style.display="None"
    document.querySelector('.text_cross_net_2_2').style.display="None"
  }
  document.querySelector(".container-col-8-1").style.display="block"
  document.querySelector('.network-1-text').style.backgroundColor="chartreuse"
  document.querySelector(".container-col-8-2").style.display="block"
  document.querySelector('.network-2-text').style.backgroundColor="chartreuse"
  document.querySelector('.btn.add-connect-btn').style.display="inline-block"
  document.querySelector('.btn.add-node-btn').style.display="inline-block"
  document.querySelector('.btn.crossover-btn').style.display="inline-block"
  get_condition="None"
  set="None"
})


document.querySelector(".crossover-btn").addEventListener("click",function(event){
  document.querySelector(".container-col-8-1").style.display="None"
  document.querySelector(".container-col-8-2").style.display="None"
  document.querySelector('#c_net_1').style.display="block"
  document.querySelector('#c_net_1_2').style.display="block"
  document.querySelector('#c_net_2').style.display="block"
  document.querySelector('#c_net_2_2').style.display="block"
  document.querySelector('#c_net_3').style.display="block"
  document.querySelector('#c_net_3_2').style.display="block"
  document.querySelector('.start_crossover').style.display="block"
  document.querySelector('.text_cross_net_1').style.display="inline-block"
  document.querySelector('.text_cross_net_2').style.display="inline-block"
  document.querySelector('.text_cross_net_1_2').style.display="inline-block"
  document.querySelector('.text_cross_net_2_2').style.display="inline-block"
  document.querySelector('.text_cross_net_3').style.display="inline-block"
  get_condition="None"
  set="None"
  active=true
})




var limit_random=8

var mp_net_1 = {}

var mp_aks_net_1 ={}

var hist_marker={}

var counter_1=1

var counter_2=1

var innove=12

var counter_1_2=1

var counter_2_2=1

var keep_track_add={}

var keep_track_connect={1:[5,6],2:[5,6],3:[5,6],4:[5,6],5:[7,8],6:[7,8]}

var keep_track_add_2={}

var keep_track_connect_2={1:[5,6],2:[5,6],3:[5,6],4:[5,6],5:[7,8],6:[7,8]}

var nodes_net_1=[1,2,3,4,5,6,7,8]

var nodes_net_2=[1,2,3,4,5,6,7,8]

var element = document.createElement("style")

document.head.appendChild(element)

var sheet=element.sheet

connections={}

mp_net_1 ={
  1:[5,6],2:[5,6],3:[5,6],4:[5,6],
  5:[7,8],6:[7,8]
}

mp_aks_net_1= {
  5:[1,2,3,4],6:[1,2,3,4],
  7:[5,6],8:[5,6]
}

mp_net_2 ={
  1:[5,6],2:[5,6],3:[5,6],4:[5,6],
  5:[7,8],6:[7,8]
}

mp_aks_net_2= {
  5:[1,2,3,4],6:[1,2,3,4],
  7:[5,6],8:[5,6]
}

hist_marker={
  1:{5:"1",6:"2"},2:{5:"3",6:"4"},3:{5:"5",6:"6"},
  4:{5:"7",6:"8"},5:{7:"9",8:"10"},
  6:{7:"11",8:"12"}
}


document.querySelector(".add-node-btn").addEventListener('click',function(event){


  // debugger
  if (set===1){
    //debugger
    var i=0
    while(true){
        i+=1
        if (i==500){
          return console.log("not found")
        }
        first_random_number=nodes_net_1[Math.floor(nodes_net_1.length*Math.random())]
        second_random_number=nodes_net_1[Math.floor(nodes_net_1.length*Math.random())]
        try{
          if (mp_net_1[first_random_number].indexOf(second_random_number)!==-1){
            if (first_random_number in keep_track_add){
              if (second_random_number in keep_track_add[fisrt_random_number]){
                continue;
              }
              else{
                break;
              }
            }
            else{
              break;
            }
          }
        }
        catch(err){
          //console.log(err)
        }
    }


    //limit_random++
    if (first_random_number in connections){
      if (second_random_number in connections[first_random_number]){
        limit = connections[first_random_number][second_random_number]
        get_main=limit
      }
      else {
        limit_random++
        p={}
        p[second_random_number]=limit_random
        connections[first_random_number]=Object.assign({},connections[first_random_number],p)
        get_main=limit_random
      }
    }
    else{
      limit_random++
      p={}
      p[second_random_number]=limit_random
      connections[first_random_number]=p
      get_main=limit_random
    }

    nodes_net_1.push(get_main)




    //show random numbers
    document.querySelector(".random_text_1").innerHTML="node_1 is <strong>"+first_random_number.toString()+"</strong>"
    document.querySelector(".random_text_2").innerHTML="node_2 is <strong>"+second_random_number.toString()+"</strong>"

    // updtae mp and mp-aks and hist-marker
    if (first_random_number in mp_net_1){
      mp_net_1[first_random_number].push(get_main)
    }
    else if (!(first_random_number in mp_net_1)){
      mp_net_1[first_random_number]=[get_main]
    }
    mp_net_1[get_main]=[second_random_number]
    var index=mp_net_1[first_random_number].indexOf(second_random_number)
    if (index>-1){
      mp_net_1[first_random_number].splice(index,1)
    }


    if (second_random_number in mp_aks_net_1){
      mp_aks_net_1[second_random_number].push(get_main)
    }
    else if (!(second_random_number in mp_aks_net_1)){
      mp_aks_net_1[second_random_number]=[get_main]
    }
    mp_aks_net_1[get_main]=[first_random_number]
    var index=mp_aks_net_1[second_random_number].indexOf(first_random_number)
    if (index>-1){
      mp_aks_net_1[second_random_number].splice(index,1)
    }


    if (first_random_number in hist_marker){
      if (get_main in hist_marker[first_random_number]){

      }
      else{
        innove++
        var passive={}
        passive[get_main]=innove.toString()
        hist_marker[first_random_number]=Object.assign({},hist_marker[first_random_number],passive)
        innove++
        passive={}
        passive[second_random_number]=innove.toString()
        if (get_main in hist_marker){
          hist_marker[get_main]=Object.assign({},hist_marker[get_main],passive)
        }
        else{
          hist_marker[get_main]=passive
        }
      }
    }
    else {
      innove++
      passive={}
      passive[get_main]=innove.toString()
      hist_marker[fisrt_random_number]=passive
      innove++
      passive={}
      passive[second_random_number]=innove.toString()
      if (get_main in hist_marker){
        hist_marker[get_main]=Object.assign({},hist_marker[get_main],passive)
      }
      else{
        hist_marker[get_main]=passive
      }
    }

    if (parseInt(hist_marker[first_random_number][get_main])>50){
      return console.log("impossible to expand")
    }

    //crossover filling
    var fill=hist_marker[first_random_number][get_main]
    fill_kamel="#c_"+fill
    document.querySelector(fill_kamel+" .crossover_text").innerHTML="<p><strong>"+"inn_"+fill+"</strong></p>"
    fill=hist_marker[get_main][second_random_number]
    fill_kamel="#c_"+fill
    document.querySelector(fill_kamel+" .crossover_text").innerHTML="<p><strong>"+"inn_"+fill+"</strong></p>"


    // update keep_track_add
    if (first_random_number in keep_track_add){
      keep_track_add[first_random_number].push(second_random_number)
    }
    else{
      keep_track_add[first_random_number]=[second_random_number]
    }


    //calculating left diff for node
    if (first_random_number>19 && first_random_number<100){
       first_random_number_t=doConvert(first_random_number).slice(0,-2)
    }
    else{
       first_random_number_t=doConvert(first_random_number).slice(0,-1)
    }
    if (second_random_number>19 && second_random_number<100){
       second_random_number_t=doConvert(second_random_number).slice(0,-2)
    }
    else{
       second_random_number_t=doConvert(second_random_number).slice(0,-1)
    }
    if (first_random_number_t.indexOf(" ", 1)!==-1){
      var place=first_random_number_t.indexOf(" ",1)
      first_random_number_t=first_random_number_t.slice(0,place)+"_"+first_random_number_t.slice(place+1)
    }
    if (second_random_number_t.indexOf(" ", 1)!==-1){
      var place=second_random_number_t.indexOf(" ",1)
      second_random_number_t=second_random_number_t.slice(0,place)+"_"+second_random_number_t.slice(place+1)
    }
    var first_node_l=document.querySelector('#'+first_random_number_t).offsetLeft
    var second_node_l=document.querySelector('#'+second_random_number_t).offsetLeft
    var diff_l=second_node_l-first_node_l
    created_l=document.querySelector('#'+first_random_number_t).offsetLeft+diff_l/2
    created_l=created_l.toString()+"px"

    //calculating top diff for node
    var first_node_h=document.querySelector('#'+first_random_number_t).offsetTop
    var second_node_h=document.querySelector('#'+second_random_number_t).offsetTop
    var diff_h=first_node_h-second_node_h
    created_h=document.querySelector('#'+second_random_number_t).offsetTop+diff_h/2
    created_h=created_h.toString()+"px"

    var stl=".new_node_"+(counter_1.toString())+" {width: 10px; height: 10px; background-color: aquamarine; border-style: groove; border-radius: 35px; position: absolute; left:" + created_l + "; top:" + created_h + "; display:block; padding-top:3px;}"

    var new_element=document.createElement("div")

    document.querySelector('.implement .container-col-8-1').appendChild(new_element)

    var get_divs=document.querySelectorAll('.implement .container-col-8-1 div')

    document.querySelectorAll('.implement .container-col-8-1 div')[get_divs.length-1].classList.add("new_node_"+(counter_1.toString()))

    var pass;

    if (get_main>19 && get_main<100){
       pass=doConvert(get_main).slice(0,-2)
    }
    else{
       pass=doConvert(get_main).slice(0,-1)
    }

    if (pass.indexOf(" ", 1)!==-1){
      var place=pass.indexOf(" ",1)
      pass=pass.slice(0,place)+"_"+pass.slice(place+1)
    }

    document.querySelectorAll('.implement .container-col-8-1 div')[get_divs.length-1].id=pass

    sheet.insertRule(stl)

    //for name inside node
    document.querySelectorAll('.implement .container-col-8-1 div')[get_divs.length-1].innerHTML="<p>"+get_main.toString()+"</p>"
    var l=document.querySelectorAll('.implement .container-col-8-1 div p').length
    document.querySelectorAll('.implement .container-col-8-1 div p')[l-1].classList.add("new_node_text")

    // change color of line
    if (first_random_number<=6 && second_random_number<=8){
      document.querySelector("."+"line_"+first_random_number.toString()+"_to_"+second_random_number.toString()).style.backgroundColor="aqua";
    }


    counter_1++

  }
  else if (set===2){
    //debugger
    var i=0
    while(true){
        i+=1
        if (i==500){
          return console.log("not found")
        }
        console.log(i)
        first_random_number=nodes_net_2[Math.floor(nodes_net_2.length*Math.random())]
        second_random_number=nodes_net_2[Math.floor(nodes_net_2.length*Math.random())]
        try{
          console.log(mp_net_2[first_random_number].indexOf(second_random_number)!==-1)
          if (mp_net_2[first_random_number].indexOf(second_random_number)!==-1){
            if (first_random_number in keep_track_add_2){
              if (second_random_number in keep_track_add_2[fisrt_random_number]){
                continue;
              }
              else{
                break;
              }
            }
            else{
              break;
            }
          }
        }
        catch(err){
          //console.log(err)
        }
    }


    //limit_random++
    if (first_random_number in connections){
      if (second_random_number in connections[first_random_number]){
        limit = connections[first_random_number][second_random_number]
        get_main=limit
      }
      else {
        limit_random++
        p={}
        p[second_random_number]=limit_random
        connections[first_random_number]=Object.assign({},connections[first_random_number],p)
        get_main=limit_random
      }
    }
    else{
      limit_random++
      p={}
      p[second_random_number]=limit_random
      connections[first_random_number]=p
      get_main=limit_random
    }

    nodes_net_2.push(get_main)

    //show random numbers
    document.querySelectorAll(".random_text_1")[1].innerHTML="node_1 is <strong>"+first_random_number.toString()+"</strong>"
    document.querySelectorAll(".random_text_2")[1].innerHTML="node_2 is <strong>"+second_random_number.toString()+"</strong>"

    // updtae mp and mp-aks and hist-marker
    if (first_random_number in mp_net_2){
      mp_net_2[first_random_number].push(get_main)
    }
    else if (!(first_random_number in mp_net_2)){
      mp_net_2[first_random_number]=[get_main]
    }
    mp_net_2[get_main]=[second_random_number]
    var index=mp_net_2[first_random_number].indexOf(second_random_number)
    if (index>-1){
      mp_net_2[first_random_number].splice(index,1)
    }


    if (second_random_number in mp_aks_net_2){
      mp_aks_net_2[second_random_number].push(get_main)
    }
    else if (!(second_random_number in mp_aks_net_2)){
      mp_aks_net_2[second_random_number]=[get_main]
    }
    mp_aks_net_2[get_main]=[first_random_number]
    var index=mp_aks_net_2[second_random_number].indexOf(first_random_number)
    if (index>-1){
      mp_aks_net_2[second_random_number].splice(index,1)
    }


    if (first_random_number in hist_marker){
      if (get_main in hist_marker[first_random_number]){

      }
      else{
        innove++
        var passive={}
        passive[get_main]=innove.toString()
        hist_marker[first_random_number]=Object.assign({},hist_marker[first_random_number],passive)
        innove++
        passive={}
        passive[second_random_number]=innove.toString()
        if (get_main in hist_marker){
          hist_marker[get_main]=Object.assign({},hist_marker[get_main],passive)
        }
        else{
          hist_marker[get_main]=passive
        }
      }
    }
    else {
      innove++
      passive={}
      passive[get_main]=innove.toString()
      hist_marker[fisrt_random_number]=passive
      innove++
      passive={}
      passive[second_random_number]=innove.toString()
      if (get_main in hist_marker){
        hist_marker[get_main]=Object.assign({},hist_marker[get_main],passive)
      }
      else{
        hist_marker[get_main]=passive
      }
    }

    if (parseInt(hist_marker[first_random_number][get_main])>50){
      return console.log("impossible to expand")
    }

    //crossover filling
    var fill=hist_marker[first_random_number][get_main]
    fill_kamel="#c_"+fill+"_2"
    document.querySelector(fill_kamel+" .crossover_text").innerHTML="<p><strong>"+"inn_"+fill+"</strong></p>"
    fill=hist_marker[get_main][second_random_number]
    fill_kamel="#c_"+fill+"_2"
    document.querySelector(fill_kamel+" .crossover_text").innerHTML="<p><strong>"+"inn_"+fill+"</strong></p>"


    // update keep_track_add
    if (first_random_number in keep_track_add_2){
      keep_track_add_2[first_random_number].push(second_random_number)
    }
    else{
      keep_track_add_2[first_random_number]=[second_random_number]
    }


    //calculating left diff for node
    if (first_random_number>19 && first_random_number<100){
       first_random_number_t=doConvert(first_random_number).slice(0,-2)
    }
    else{
       first_random_number_t=doConvert(first_random_number).slice(0,-1)
    }
    if (second_random_number>19 && second_random_number<100){
       second_random_number_t=doConvert(second_random_number).slice(0,-2)
    }
    else{
       second_random_number_t=doConvert(second_random_number).slice(0,-1)
    }
    if (first_random_number_t.indexOf(" ", 1)!==-1){
      var place=first_random_number_t.indexOf(" ",1)
      first_random_number_t=first_random_number_t.slice(0,place)+"_"+first_random_number_t.slice(place+1)
    }
    if (second_random_number_t.indexOf(" ", 1)!==-1){
      var place=second_random_number_t.indexOf(" ",1)
      second_random_number_t=second_random_number_t.slice(0,place)+"_"+second_random_number_t.slice(place+1)
    }
    var first_node_l=document.querySelector('#'+first_random_number_t+"_2").offsetLeft
    var second_node_l=document.querySelector('#'+second_random_number_t+"_2").offsetLeft
    var diff_l=second_node_l-first_node_l
    created_l=document.querySelector('#'+first_random_number_t+"_2").offsetLeft+diff_l/2
    created_r=parseInt(getComputedStyle(document.querySelector(".container-col-8-2")).width)-created_l-parseInt(getComputedStyle(document.querySelector('#'+first_random_number_t+"_2")).width)
    created_r=created_r.toString()+"px"
    created_l=created_l.toString()+"px"

    //calculating top diff for node
    var first_node_h=document.querySelector('#'+first_random_number_t+"_2").offsetTop
    var second_node_h=document.querySelector('#'+second_random_number_t+"_2").offsetTop
    var diff_h=first_node_h-second_node_h
    created_h=document.querySelector('#'+second_random_number_t+"_2").offsetTop+diff_h/2
    created_h=created_h.toString()+"px"

    var stl=".new_node_"+(counter_1_2.toString())+"_net_2" + " {width: 10px; height: 10px; background-color: aquamarine; border-style: groove; border-radius: 35px; position: absolute; right:" + created_r + "; top:" + created_h + "; display:block; padding-top:3px;}"

    var new_element=document.createElement("div")

    document.querySelector('.implement .container-col-8-2').appendChild(new_element)

    var get_divs=document.querySelectorAll('.implement .container-col-8-2 div')

    document.querySelectorAll('.implement .container-col-8-2 div')[get_divs.length-1].classList.add("new_node_"+(counter_1_2.toString())+"_net_2")

    var pass;

    if (get_main>19 && get_main<100){
       pass=doConvert(get_main).slice(0,-2)
    }
    else{
       pass=doConvert(get_main).slice(0,-1)
    }
    if (pass.indexOf(" ", 1)!==-1){
      var place=pass.indexOf(" ",1)
      pass=pass.slice(0,place)+"_"+pass.slice(place+1)
    }

    pass=pass+"_2"

    document.querySelectorAll('.implement .container-col-8-2 div')[get_divs.length-1].id=pass

    sheet.insertRule(stl)

    //for name inside node
    document.querySelectorAll('.implement .container-col-8-2 div')[get_divs.length-1].innerHTML="<p>"+get_main.toString()+"</p>"
    var l=document.querySelectorAll('.implement .container-col-8-2 div p').length
    document.querySelectorAll('.implement .container-col-8-2 div p')[l-1].classList.add("new_node_text")

    // change color of line
    if (first_random_number<=6 && second_random_number<=8){
      document.querySelector("."+"line_2_"+first_random_number.toString()+"_to_"+second_random_number.toString()).style.backgroundColor="aqua";
    }


    counter_1_2++

  }


})





document.querySelector(".add-connect-btn").addEventListener('click',function(event){


  if (set===1){

    // debugger
    var k=0
    while(true){
        k++
        if (k==200){
          return console.log("not found")
        }
        first_random_number=nodes_net_1[Math.floor(nodes_net_1.length*Math.random())]
        second_random_number=nodes_net_1[Math.floor(nodes_net_1.length*Math.random())]
        var valid=validity(first_random_number,second_random_number,mp_aks_net_1)
        if (valid==true && mp_aks_net_1[second_random_number].indexOf(first_random_number)==-1){
          if (first_random_number in keep_track_connect){
            if (keep_track_connect[first_random_number].indexOf(second_random_number)!==-1){
              continue
            }
            else{
              break
            }
          }
          else{
            break
          }
        }
    }


    // updtae mp and mp-aks and hist-marker
    if (first_random_number in mp_net_1){
      mp_net_1[first_random_number].push(second_random_number)
    }
    else if (!(first_random_number in mp_net_1)){
      mp_net_1[first_random_number]=[second_random_number]
    }



    if (second_random_number in mp_aks_net_1){
      mp_aks_net_1[second_random_number].push(first_random_number)
    }
    else if (!(second_random_number in mp_aks_net_1)){
      mp_aks_net_1[second_random_number]=[first_random_number]
    }



    if (first_random_number in hist_marker){
      if (second_random_number in hist_marker[first_random_number]){

      }
      else{
        var passive={}
        innove++
        passive[second_random_number]=innove.toString()
        hist_marker[first_random_number]=Object.assign({},hist_marker[first_random_number],passive)
      }
    }
    else {
      innove++
      passive={}
      passive[second_random_number]=innove.toString()
      hist_marker[fisrt_random_number]=passive
    }


    if (parseInt(hist_marker[first_random_number][second_random_number])>50){
      return console.log("impossible to expand")
    }

    //crossover filling
    var fill=hist_marker[first_random_number][second_random_number]
    fill_kamel="#c_"+fill
    document.querySelector(fill_kamel+" .crossover_text").innerHTML="<p><strong>"+"inn_"+fill+"</strong></p>"


    //show random numbers
    document.querySelector(".random_text_1").innerHTML="node_1 is <strong>"+first_random_number.toString()+"</strong>"
    document.querySelector(".random_text_2").innerHTML="node_2 is <strong>"+second_random_number.toString()+"</strong>"

    //calculating top diff for line
    if (first_random_number>19 && first_random_number<100){
       first_random_number_t=doConvert(first_random_number).slice(0,-2)
    }
    else{
       first_random_number_t=doConvert(first_random_number).slice(0,-1)
    }
    if (second_random_number>19 && second_random_number<100){
       second_random_number_t=doConvert(second_random_number).slice(0,-2)
    }
    else{
       second_random_number_t=doConvert(second_random_number).slice(0,-1)
    }
    if (first_random_number_t.indexOf(" ", 1)!==-1){
      var place=first_random_number_t.indexOf(" ",1)
      first_random_number_t=first_random_number_t.slice(0,place)+"_"+first_random_number_t.slice(place+1)
    }
    if (second_random_number_t.indexOf(" ", 1)!==-1){
      var place=second_random_number_t.indexOf(" ",1)
      second_random_number_t=second_random_number_t.slice(0,place)+"_"+second_random_number_t.slice(place+1)
    }
    var first_node_h=document.querySelector('#'+first_random_number_t).offsetTop
    var second_node_h=document.querySelector('#'+second_random_number_t).offsetTop
    var diff_h=first_node_h-(second_node_h)
    var diff_h_n=diff_h-parseInt(getComputedStyle(document.querySelector('#'+second_random_number_t)).width)


    //calculating left diff for line
    var first_node_l=document.querySelector('#'+first_random_number_t).offsetLeft
    var second_node_l=document.querySelector('#'+second_random_number_t).offsetLeft
    var diff_l=second_node_l-first_node_l

    //calculating height for line
    var created_h=Math.sqrt(diff_h_n**2 + diff_l**2)

    //calculating degree for line
    var created_d=Math.atan(diff_l/diff_h_n)
    created_d=(created_d*90)/1.5707963267948966
    created_d=created_d.toString()

    //calculating left for line
    var created_l=document.querySelector("#"+first_random_number_t).offsetLeft+(parseInt(getComputedStyle(document.querySelector("#"+first_random_number_t)).width)/2)
    created_l=created_l.toString()+"px"

    //calculating top for line
    var created_t=document.querySelector("#"+first_random_number_t).offsetTop-created_h
    created_h=created_h.toString()+"px"
    created_t=created_t.toString()+"px"

    var stl=".new_line_"+counter_2.toString()+" {width: 2px; height:" + created_h + " ; background-color: red; top: " + created_t + "; position: absolute;  left: " +  created_l  +" ;  transform: rotate(" + created_d + "deg)" + "; transform-origin: 50% 100%; position: absolute;}"

    var new_element=document.createElement("div")

    document.querySelector('.implement .container-col-8-1').appendChild(new_element)

    var get_divs=document.querySelectorAll('.implement .container-col-8-1 div')

    document.querySelectorAll('.implement .container-col-8-1 div')[get_divs.length-1].classList.add("new_line_"+(counter_2.toString()))

    sheet.insertRule(stl)

    counter_2++

    // update keep_track_add
    if (first_random_number in keep_track_connect){
      keep_track_connect[first_random_number].push(second_random_number)
    }
    else{
      keep_track_connect[first_random_number]=[second_random_number]
    }

  }
  else if (set===2){
    // debugger
    var k=0
    while(true){
        k++
        if (k==200){
          return console.log("not found")
        }
        first_random_number=nodes_net_2[Math.floor(nodes_net_2.length*Math.random())]
        second_random_number=nodes_net_2[Math.floor(nodes_net_2.length*Math.random())]
        var valid=validity(first_random_number,second_random_number,mp_aks_net_2)
        if (valid==true && mp_aks_net_2[second_random_number].indexOf(first_random_number)==-1){
          if (first_random_number in keep_track_connect_2){
            if (keep_track_connect_2[first_random_number].indexOf(second_random_number)!==-1){
              continue
            }
            else{
              break
            }
          }
          else{
            break
          }
        }
    }

    // updtae mp and mp-aks and hist-marker
    if (first_random_number in mp_net_2){
      mp_net_2[first_random_number].push(second_random_number)
    }
    else if (!(first_random_number in mp_net_2)){
      mp_net_2[first_random_number]=[second_random_number]
    }



    if (second_random_number in mp_aks_net_2){
      mp_aks_net_2[second_random_number].push(first_random_number)
    }
    else if (!(second_random_number in mp_aks_net_2)){
      mp_aks_net_2[second_random_number]=[first_random_number]
    }



    if (first_random_number in hist_marker){
      if (second_random_number in hist_marker[first_random_number]){

      }
      else{
        var passive={}
        innove++
        passive[second_random_number]=innove.toString()
        hist_marker[first_random_number]=Object.assign({},hist_marker[first_random_number],passive)
      }
    }
    else {
      innove++
      passive={}
      passive[second_random_number]=innove.toString()
      hist_marker[fisrt_random_number]=passive
    }


    if (parseInt(hist_marker[first_random_number][second_random_number])>50){
      return console.log("impossible to expand")
    }

    //crossover filling
    var fill=hist_marker[first_random_number][second_random_number]
    fill_kamel="#c_"+fill+"_2"
    document.querySelector(fill_kamel+" .crossover_text").innerHTML="<p><strong>"+"inn_"+fill+"</strong></p>"

    //show random numbers
    document.querySelectorAll(".random_text_1")[1].innerHTML="node_1 is <strong>"+first_random_number.toString()+"</strong>"
    document.querySelectorAll(".random_text_2")[1].innerHTML="node_2 is <strong>"+second_random_number.toString()+"</strong>"

    //calculating top diff for line
    if (first_random_number>19 && first_random_number<100){
       first_random_number_t=doConvert(first_random_number).slice(0,-2)
    }
    else{
       first_random_number_t=doConvert(first_random_number).slice(0,-1)
    }
    if (second_random_number>19 && second_random_number<100){
       second_random_number_t=doConvert(second_random_number).slice(0,-2)
    }
    else{
       second_random_number_t=doConvert(second_random_number).slice(0,-1)
    }
    if (first_random_number_t.indexOf(" ", 1)!==-1){
      var place=first_random_number_t.indexOf(" ",1)
      first_random_number_t=first_random_number_t.slice(0,place)+"_"+first_random_number_t.slice(place+1)
    }
    if (second_random_number_t.indexOf(" ", 1)!==-1){
      var place=second_random_number_t.indexOf(" ",1)
      second_random_number_t=second_random_number_t.slice(0,place)+"_"+second_random_number_t.slice(place+1)
    }
    var first_node_h=document.querySelector('#'+first_random_number_t+"_2").offsetTop
    var second_node_h=document.querySelector('#'+second_random_number_t+"_2").offsetTop
    var diff_h=first_node_h-(second_node_h)
    var diff_h_n=diff_h-parseInt(getComputedStyle(document.querySelector('#'+second_random_number_t+"_2")).width)


    //calculating left diff for line
    var first_node_l=document.querySelector('#'+first_random_number_t+"_2").offsetLeft
    var second_node_l=document.querySelector('#'+second_random_number_t+"_2").offsetLeft
    var diff_l=second_node_l-first_node_l

    //calculating height for line
    var created_h=Math.sqrt(diff_h_n**2 + diff_l**2)

    //calculating degree for line
    var created_d=Math.atan(diff_l/diff_h_n)
    created_d=(created_d*90)/1.5707963267948966
    created_d=created_d.toString()

    //calculating left for line
    var created_l=document.querySelector("#"+first_random_number_t+"_2").offsetLeft+(parseInt(getComputedStyle(document.querySelector("#"+first_random_number_t+"_2")).width)/2)
    created_r=parseInt(getComputedStyle(document.querySelector(".container-col-8-2")).width)-created_l-2
    created_r=created_r.toString()+"px"
    created_l=created_l.toString()+"px"

    //calculating top for line
    var created_t=document.querySelector("#"+first_random_number_t+"_2").offsetTop-created_h
    created_h=created_h.toString()+"px"
    created_t=created_t.toString()+"px"

    var stl=".new_line_"+counter_2_2.toString()+"_net_2"+" {width: 2px; height:" + created_h + " ; background-color: red; top: " + created_t + "; position: absolute;  right: " +  created_r  +" ;  transform: rotate(" + created_d + "deg)" + "; transform-origin: 50% 100%; position: absolute;}"

    var new_element=document.createElement("div")

    document.querySelector('.implement .container-col-8-2').appendChild(new_element)

    var get_divs=document.querySelectorAll('.implement .container-col-8-2 div')

    document.querySelectorAll('.implement .container-col-8-2 div')[get_divs.length-1].classList.add("new_line_"+(counter_2_2.toString())+"_net_2")

    sheet.insertRule(stl)

    counter_2_2++

    // update keep_track_add
    if (first_random_number in keep_track_connect_2){
      keep_track_connect_2[first_random_number].push(second_random_number)
    }
    else{
      keep_track_connect_2[first_random_number]=[second_random_number]
    }


  }


})



//crossover part
document.querySelector(".start_crossover").addEventListener("click",function(event){
  //debugger
  var fitness_1 = Math.floor(500*Math.random())
  var fitness_2 = Math.floor(500*Math.random())
  document.querySelector(".text_cross_net_1_2").innerText="fitness is: "+fitness_1.toString()
  document.querySelector(".text_cross_net_2_2").innerText="fitness is: "+fitness_2.toString()
  match=0
  for (var i=1 ; i<=50 ; i++){
    var random_number=Math.floor(1+2*Math.random())
    txt_1=document.querySelector("#c_"+i.toString()+" .crossover_text").innerText
    txt_2=document.querySelector("#c_"+i.toString()+"_2"+" .crossover_text").innerText
    if(txt_1==txt_2 && txt_1[0]=="i" && i==match+1){
      match++
      if (random_number===1){
        document.querySelector("#c_"+i.toString()).style.backgroundColor="green"
        document.querySelector("#c_"+i.toString()+"_2").style.backgroundColor="red"
        document.querySelector("#r_"+i.toString()+" .crossover_text").innerHTML="<p><strong>" + txt_1 + "</strong></p>"
        document.querySelector("#r_"+i.toString()).style.backgroundColor="green"
      }
      else if (random_number===2){
        document.querySelector("#c_"+i.toString()+"_2").style.backgroundColor="green"
        document.querySelector("#c_"+i.toString()).style.backgroundColor="red"
        document.querySelector("#r_"+i.toString()+" .crossover_text").innerHTML="<p><strong>" + txt_2 + "</strong></p>"
        document.querySelector("#r_"+i.toString()).style.backgroundColor="green"
      }
    }
    else{
      if (fitness_1>fitness_2){
        if (txt_1[0]=="i"){
          document.querySelector("#c_"+i.toString()).style.backgroundColor="green"
          document.querySelector("#c_"+i.toString()+"_2").style.backgroundColor="red"
          document.querySelector("#r_"+i.toString()+" .crossover_text").innerHTML="<p><strong>" + txt_1 + "</strong></p>"
          document.querySelector("#r_"+i.toString()).style.backgroundColor="green"
        }
        else{
          document.querySelector("#c_"+i.toString()).style.backgroundColor="red"
          document.querySelector("#c_"+i.toString()+"_2").style.backgroundColor="red"
        }
      }
      else{
        if (txt_2[0]=="i"){
          document.querySelector("#c_"+i.toString()+"_2").style.backgroundColor="green"
          document.querySelector("#c_"+i.toString()).style.backgroundColor="red"
          document.querySelector("#r_"+i.toString()+" .crossover_text").innerHTML="<p><strong>" + txt_2 + "</strong></p>"
          document.querySelector("#r_"+i.toString()).style.backgroundColor="green"
        }
        else{
          document.querySelector("#c_"+i.toString()).style.backgroundColor="red"
          document.querySelector("#c_"+i.toString()+"_2").style.backgroundColor="red"
        }
      }
    }
  }
})




document.querySelector(".col-6 .btn.btn-css-2").addEventListener("click",function(){
  document.querySelector(".q_mix_base .q_mix_l").style.display="None"
  document.querySelector(".q_mix_base .q_mix_r").style.display="None"
  document.querySelector(".main .row.q_mix_base").style.height="650px"
})





function validity(x,y,mp_aks){
  //debugger
  if (y==1 || y==2 || y==3 || y==4){
    return false
  }
  function recursive(x,y,mp_aks){
    var bool=false
    var counter=0
    var containter=mp_aks[y]
    for (var i=0; i<containter.length; i++){
      if (containter[i]==x){
        return true
      }
      else if (containter[i]!==x){
        if (containter[i]==1 || containter[i]==2 || containter[i]==3 || containter[i]==4){
          counter+=1
        }
        else{
          number=containter[i]
          bool=recursive(x,number,mp_aks)
          counter+=1
        }
        if (bool==true){
          return true
        }
        if (counter==containter.length){
          return false
        }
      }
    }
  }
  return recursive(x,y,mp_aks)
}








function doConvert (numberInput){

    let oneToTwenty = ['','one ','two ','three ','four ', 'five ','six ','seven ','eight ','nine ','ten ',
    'eleven ','twelve ','thirteen ','fourteen ','fifteen ','sixteen ','seventeen ','eighteen ','nineteen '];
    let tenth = ['', '', 'twenty','thirty','forty','fifty', 'sixty','seventy','eighty','ninety'];

    if(numberInput.toString().length > 7) return console.log(overflow) ;
        //let num = ('0000000000'+ numberInput).slice(-10).match(/^(\d{1})(\d{2})(\d{2})(\d{2})(\d{1})(\d{2})$/);
  let num = ('0000000'+ numberInput).slice(-7).match(/^(\d{1})(\d{1})(\d{2})(\d{1})(\d{2})$/);
    if(!num) return;

    let outputText = num[1] != 0 ? (oneToTwenty[Number(num[1])] || `${tenth[num[1][0]]} ${oneToTwenty[num[1][1]]}` )+' million ' : '';

    outputText +=num[2] != 0 ? (oneToTwenty[Number(num[2])] || `${tenth[num[2][0]]} ${oneToTwenty[num[2][1]]}` )+'hundred ' : '';
    outputText +=num[3] != 0 ? (oneToTwenty[Number(num[3])] || `${tenth[num[3][0]]} ${oneToTwenty[num[3][1]]}`)+' thousand ' : '';
    outputText +=num[4] != 0 ? (oneToTwenty[Number(num[4])] || `${tenth[num[4][0]]} ${oneToTwenty[num[4][1]]}`) +'hundred ': '';
    outputText +=num[5] != 0 ? (oneToTwenty[Number(num[5])] || `${tenth[num[5][0]]} ${oneToTwenty[num[5][1]]} `) : '';

    return outputText
}



function calculate_distance_net_1(node_1,node_2,doConvert){
  debugger
  if (node_1>19 && node_1<100){
     node_1_t=doConvert(node_1).slice(0,-2)
  }
  else{
     node_1_t=doConvert(node_1).slice(0,-1)
  }
  if (node_2>19 && node_2<100){
     node_2_t=doConvert(node_2).slice(0,-2)
  }
  else{
     node_2_t=doConvert(node_2).slice(0,-1)
  }
  //for nodes
  var first_node_l=document.querySelector('#'+node_1_t).offsetLeft
  var second_node_l=document.querySelector('#'+node_2_t).offsetLeft
  var diff_l=second_node_l-first_node_l
  created_l=document.querySelector('#'+node_1_t).offsetLeft+diff_l/2
  created_l=created_l.toString()+"px"


  var first_node_h=document.querySelector('#'+node_1_t).offsetTop
  var second_node_h=document.querySelector('#'+node_2_t).offsetTop
  var diff_h=first_node_h-second_node_h
  created_h=document.querySelector('#'+node_2_t).offsetTop+diff_h/2
  created_h=created_h.toString()+"px"

  console.log(created_l)
  console.log(created_h)

  //for lines
  //calculating top diff for line
  var first_node_h=document.querySelector('#'+node_1_t).offsetTop
  var second_node_h=document.querySelector('#'+node_2_t).offsetTop
  var diff_h=first_node_h-(second_node_h)
  var diff_h_n=diff_h-parseInt(getComputedStyle(document.querySelector('#'+node_2_t)).width)


  //calculating left diff for line
  var first_node_l=document.querySelector('#'+node_1_t).offsetLeft
  var second_node_l=document.querySelector('#'+node_2_t).offsetLeft
  var diff_l=second_node_l-first_node_l

  //calculating height for line
  var created_h=Math.sqrt(diff_h_n**2 + diff_l**2)

  //calculating degree for line
  var created_d=Math.atan(diff_l/diff_h_n)
  created_d=(created_d*90)/1.5707963267948966
  created_d=created_d.toString()

  //calculating left for line
  var created_l=document.querySelector("#"+node_1_t).offsetLeft+(parseInt(getComputedStyle(document.querySelector("#"+node_1_t)).width)/2)
  created_l=created_l.toString()+"px"

  //calculating top for line
  var created_t=document.querySelector("#"+node_1_t).offsetTop-created_h
  created_h=created_h.toString()+"px"
  created_t=created_t.toString()+"px"

  console.log(created_l)
  console.log(created_h)
  console.log(created_t)
  console.log(created_d)

}

function calculate_distance_net_2(node_1,node_2,doConvert){
  debugger
  if (node_1>19 && node_1<100){
     node_1_t=doConvert(node_1).slice(0,-2)
  }
  else{
     node_1_t=doConvert(node_1).slice(0,-1)
  }
  if (node_2>19 && node_2<100){
     node_2_t=doConvert(node_2).slice(0,-2)
  }
  else{
     node_2_t=doConvert(node_2).slice(0,-1)
  }
  //for nodes
  var first_node_l=document.querySelector('#'+node_1_t+"_2").offsetLeft
  var second_node_l=document.querySelector('#'+node_2_t+"_2").offsetLeft
  var diff_l=second_node_l-first_node_l
  created_l=document.querySelector('#'+node_1_t+"_2").offsetLeft+diff_l/2
  created_l=created_l.toString()+"px"


  var first_node_h=document.querySelector('#'+node_1_t+"_2").offsetTop
  var second_node_h=document.querySelector('#'+node_2_t+"_2").offsetTop
  var diff_h=first_node_h-second_node_h
  created_h=document.querySelector('#'+node_2_t+"_2").offsetTop+diff_h/2
  created_h=created_h.toString()+"px"

  console.log(created_l)
  console.log(created_h)

  //for lines
  //calculating top diff for line
  var first_node_h=document.querySelector('#'+node_1_t+"_2").offsetTop
  var second_node_h=document.querySelector('#'+node_2_t+"_2").offsetTop
  var diff_h=first_node_h-(second_node_h)
  var diff_h_n=diff_h-parseInt(getComputedStyle(document.querySelector('#'+node_2_t+"_2")).width)


  //calculating left diff for line
  var first_node_l=document.querySelector('#'+node_1_t+"_2").offsetLeft
  var second_node_l=document.querySelector('#'+node_2_t+"_2").offsetLeft
  var diff_l=second_node_l-first_node_l

  //calculating height for line
  var created_h=Math.sqrt(diff_h_n**2 + diff_l**2)

  //calculating degree for line
  var created_d=Math.atan(diff_l/diff_h_n)
  created_d=(created_d*90)/1.5707963267948966
  created_d=created_d.toString()

  //calculating left for line
  var created_l=document.querySelector("#"+node_1_t+"_2").offsetLeft+(parseInt(getComputedStyle(document.querySelector("#"+node_1_t+"_2")).width)/2)
  created_l=created_l.toString()+"px"

  //calculating top for line
  var created_t=document.querySelector("#"+node_1_t+"_2").offsetTop-created_h
  created_h=created_h.toString()+"px"
  created_t=created_t.toString()+"px"

  console.log(created_l)
  console.log(created_h)
  console.log(created_t)
  console.log(created_d)

}


//____________________________________________________________________________________________________

var env_active=0

var fill_buffer=0

var train=0

var count_fun=1

var passive_variable=0

var on_off="off"

var get_obs=0

var get_l_a=0

var get_h=0

var get_input=0

var get_q=0

var get_l_s=0

var get_state=0

var step=1

o_dict={1:{"agent_1":[],"agent_2":[]},2:{"agent_1":[],"agent_2":[]},3:{"agent_1":[],"agent_2":[]},4:{"agent_1":[],"agent_2":[]},
5:{"agent_1":[],"agent_2":[]},6:{"agent_1":[],"agent_2":[]},7:{"agent_1":[],"agent_2":[]},8:{"agent_1":[],"agent_2":[]},9:{"agent_1":[],"agent_2":[]},10:{"agent_1":[],"agent_2":[]}}

l_dict={1:{"agent_1":[],"agent_2":[]},2:{"agent_1":[],"agent_2":[]},3:{"agent_1":[],"agent_2":[]},4:{"agent_1":[],"agent_2":[]},
5:{"agent_1":[],"agent_2":[]},6:{"agent_1":[],"agent_2":[]},7:{"agent_1":[],"agent_2":[]},8:{"agent_1":[],"agent_2":[]},9:{"agent_1":[],"agent_2":[]},10:{"agent_1":[],"agent_2":[]}}


h_dict={1:{"agent_1":[],"agent_2":[]},2:{"agent_1":[],"agent_2":[]},3:{"agent_1":[],"agent_2":[]},4:{"agent_1":[],"agent_2":[]},
5:{"agent_1":[],"agent_2":[]},6:{"agent_1":[],"agent_2":[]},7:{"agent_1":[],"agent_2":[]},8:{"agent_1":[],"agent_2":[]},9:{"agent_1":[],"agent_2":[]},10:{"agent_1":[],"agent_2":[]}}

q_dict={1:{"agent_1":[],"agent_2":[]},2:{"agent_1":[],"agent_2":[]},3:{"agent_1":[],"agent_2":[]},4:{"agent_1":[],"agent_2":[]},
5:{"agent_1":[],"agent_2":[]},6:{"agent_1":[],"agent_2":[]},7:{"agent_1":[],"agent_2":[]},8:{"agent_1":[],"agent_2":[]},9:{"agent_1":[],"agent_2":[]},10:{"agent_1":[],"agent_2":[]}}

a_dict={1:{"agent_1":[],"agent_2":[]},2:{"agent_1":[],"agent_2":[]},3:{"agent_1":[],"agent_2":[]},4:{"agent_1":[],"agent_2":[]},
5:{"agent_1":[],"agent_2":[]},6:{"agent_1":[],"agent_2":[]},7:{"agent_1":[],"agent_2":[]},8:{"agent_1":[],"agent_2":[]},9:{"agent_1":[],"agent_2":[]},10:{"agent_1":[],"agent_2":[]}}


hidden_dict={1:{"agent_1":[],"agent_2":[]},2:{"agent_1":[],"agent_2":[]},3:{"agent_1":[],"agent_2":[]},4:{"agent_1":[],"agent_2":[]},
5:{"agent_1":[],"agent_2":[]},6:{"agent_1":[],"agent_2":[]},7:{"agent_1":[],"agent_2":[]},8:{"agent_1":[],"agent_2":[]},9:{"agent_1":[],"agent_2":[]},10:{"agent_1":[],"agent_2":[]}}

document.querySelector(".q_mix_r .btn-css-2").addEventListener("click",function(event){
  document.querySelector(".implement_q_mix").style.display="block"
  document.querySelector('.q_mix_base').style.dispaly="none"
  if (implementation===1){
    document.querySelector('.implement_q_mix').style.top="-1500px"
  }
  implementation_2=1
})

document.querySelector(".btn-env").addEventListener("click",function(event){
  if (fill_buffer==1){
    document.querySelector(".fill-buffer").style.display="none"
    fill_buffer=0
  }
  if (train==1){
    document.querySelector(".train_part").style.display="none"
    train=0
  }
  document.querySelector(".env_base").style.display="block"
  env_active=1
}
)


document.querySelector(".btn-fill-buffer").addEventListener("click",function(event){
  if (env_active==1){
    document.querySelector(".env_base").style.display="none"
    env_active=0
  }
  if (train==1){
    document.querySelector(".train_part").style.display="none"
    train=0
  }
  document.querySelector(".fill-buffer").style.display="block"
  fill_buffer=1
})

document.querySelector(".btn-train").addEventListener("click",function(event){
  if (env_active==1){
    document.querySelector(".env_base").style.display="none"
    env_active=0
  }
  if (fill_buffer==1){
    document.querySelector(".fill-buffer").style.display="none"
    fill_buffer=0
  }
  document.querySelector(".train_part").style.display="block"
  train=1
})


document.querySelector(".step-show .btn").addEventListener("click",function(event){
  passive_variable=document.querySelectorAll(".btn-apply-fun")
  passive_variable[0].style.display="block"
})


// for getting obs
document.querySelectorAll(".btn-apply-fun")[0].addEventListener("click",function(event){
  if(get_obs==0){
    disp(get_obs,get_l_a,get_h,get_input,get_q,get_l_s)
    document.querySelector(".agent-1-obs").style.display="block"
    document.querySelector(".agent-2-obs").style.display="block"
    document.querySelector(".agent-1-state").style.display="block"
    document.querySelector(".agent-2-state").style.display="block"
    document.querySelector(".agent-state-total").style.display="block"
    if (step===1){
      var pass_o=[]
      for (var i=0; i<4; i++){
        pass_o.push(2*Math.random().toFixed(1))
        o_dict[step]["agent_1"].push(pass_o[i])
      }
      for (var i=0; i<4; i++){
        pass_o.push(2*Math.random().toFixed(1))
        o_dict[step]["agent_2"].push(pass_o[i+4])
      }
      document.querySelector(".agent-1-obs .o_first_a_1 p").innerText=pass_o[0].toString()
      document.querySelector(".agent-1-obs .o_2_a_1 p").innerText=pass_o[1].toString()
      document.querySelector(".agent-1-obs .o_3_a_1 p").innerText=pass_o[2].toString()
      document.querySelector(".agent-1-obs .o_end_a_1 p").innerText=pass_o[3].toString()
      document.querySelector(".agent-2-obs .o_first_a_2 p").innerText=pass_o[4].toString()
      document.querySelector(".agent-2-obs .o_2_a_2 p").innerText=pass_o[5].toString()
      document.querySelector(".agent-2-obs .o_3_a_2 p").innerText=pass_o[6].toString()
      document.querySelector(".agent-2-obs .o_end_a_2 p").innerText=pass_o[7].toString()
      document.querySelector(".agent-1-state .s_1_a_1 p").innerText=pass_o[0].toString()
      document.querySelector(".agent-1-state .s_2_a_1 p").innerText=pass_o[1].toString()
      document.querySelector(".agent-1-state .s_3_a_1 p").innerText=pass_o[2].toString()
      document.querySelector(".agent-1-state .s_4_a_1 p").innerText=pass_o[3].toString()
      document.querySelector(".agent-2-state .s_1_a_2 p").innerText=pass_o[4].toString()
      document.querySelector(".agent-2-state .s_2_a_2 p").innerText=pass_o[5].toString()
      document.querySelector(".agent-2-state .s_3_a_2 p").innerText=pass_o[6].toString()
      document.querySelector(".agent-2-state .s_4_a_2 p").innerText=pass_o[7].toString()
      document.querySelector(".agent-state-total .s_1 p").innerText=pass_o[0].toString()
      document.querySelector(".agent-state-total .s_2 p").innerText=pass_o[1].toString()
      document.querySelector(".agent-state-total .s_3 p").innerText=pass_o[2].toString()
      document.querySelector(".agent-state-total .s_4 p").innerText=pass_o[7].toString()
      get_obs=1
      get_state=1
      on_off="on"
    }
    else{
      document.querySelector(".agent-1-obs .o_first_a_1 p").innerText=o_dict[step]["agent_1"][0].toString()
      document.querySelector(".agent-1-obs .o_2_a_1 p").innerText=o_dict[step]["agent_1"][1].toString()
      document.querySelector(".agent-1-obs .o_3_a_1 p").innerText=o_dict[step]["agent_1"][2].toString()
      document.querySelector(".agent-1-obs .o_end_a_1 p").innerText=o_dict[step]["agent_1"][3].toString()
      document.querySelector(".agent-2-obs .o_first_a_2 p").innerText=o_dict[step]["agent_2"][0].toString()
      document.querySelector(".agent-2-obs .o_2_a_2 p").innerText=o_dict[step]["agent_2"][1].toString()
      document.querySelector(".agent-2-obs .o_3_a_2 p").innerText=o_dict[step]["agent_2"][2].toString()
      document.querySelector(".agent-2-obs .o_end_a_2 p").innerText=o_dict[step]["agent_2"][3].toString()
      document.querySelector(".agent-1-state .s_1_a_1 p").innerText=o_dict[step]["agent_1"][0].toString()
      document.querySelector(".agent-1-state .s_2_a_1 p").innerText=o_dict[step]["agent_1"][1].toString()
      document.querySelector(".agent-1-state .s_3_a_1 p").innerText=o_dict[step]["agent_1"][2].toString()
      document.querySelector(".agent-1-state .s_4_a_1 p").innerText=o_dict[step]["agent_1"][3].toString()
      document.querySelector(".agent-2-state .s_1_a_2 p").innerText=o_dict[step]["agent_2"][0].toString()
      document.querySelector(".agent-2-state .s_2_a_2 p").innerText=o_dict[step]["agent_2"][1].toString()
      document.querySelector(".agent-2-state .s_3_a_2 p").innerText=o_dict[step]["agent_2"][2].toString()
      document.querySelector(".agent-2-state .s_4_a_2 p").innerText=o_dict[step]["agent_2"][3].toString()
      document.querySelector(".agent-state-total .s_1 p").innerText=o_dict[step]["agent_1"][0].toString()
      document.querySelector(".agent-state-total .s_2 p").innerText=o_dict[step]["agent_1"][1].toString()
      document.querySelector(".agent-state-total .s_3 p").innerText=o_dict[step]["agent_1"][2].toString()
      document.querySelector(".agent-state-total .s_4 p").innerText=o_dict[step]["agent_2"][3].toString()
      get_obs=1
      get_state=1
      on_off="on"
    }
  }
  else{
    disp(get_obs,get_l_a,get_h,get_input,get_q,get_l_s,get_state)
    document.querySelector(".agent-1-obs").style.display="block"
    document.querySelector(".agent-2-obs").style.display="block"
    document.querySelector(".agent-1-state").style.display="block"
    document.querySelector(".agent-2-state").style.display="block"
    document.querySelector(".agent-state-total").style.display="block"
  }
})

// for getting last action
document.querySelectorAll(".btn-apply-fun")[1].addEventListener("click",function(event){
  if(get_l_a==0){
    disp(get_obs,get_l_a,get_h,get_input,get_q,get_l_s,get_state)
    document.querySelector(".agent-1-last_action").style.display="block"
    document.querySelector(".agent-2-last_action").style.display="block"
    if (step===1){
      sefr=0
      yek=1
      var pass_l=[]
      for (var i=0; i<5; i++){
        pass_l.push(sefr)
        l_dict[step]["agent_1"].push(pass_l[i])
      }
      for (var i=0; i<5; i++){
        pass_l.push(sefr)
        l_dict[step]["agent_2"].push(pass_l[i+5])
      }
      document.querySelector(".agent-1-last_action .l_1_a_1 p").innerText=pass_l[0].toString()
      document.querySelector(".agent-1-last_action .l_2_a_1 p").innerText=pass_l[1].toString()
      document.querySelector(".agent-1-last_action .l_3_a_1 p").innerText=pass_l[2].toString()
      document.querySelector(".agent-1-last_action .l_4_a_1 p").innerText=pass_l[3].toString()
      document.querySelector(".agent-1-last_action .l_5_a_1 p").innerText=pass_l[4].toString()
      document.querySelector(".agent-2-last_action .l_1_a_2 p").innerText=pass_l[5].toString()
      document.querySelector(".agent-2-last_action .l_2_a_2 p").innerText=pass_l[6].toString()
      document.querySelector(".agent-2-last_action .l_3_a_2 p").innerText=pass_l[7].toString()
      document.querySelector(".agent-2-last_action .l_4_a_2 p").innerText=pass_l[8].toString()
      document.querySelector(".agent-2-last_action .l_5_a_2 p").innerText=pass_l[9].toString()
      get_l_a=1
      on_off="on"
    }
    else{
      yek=1
      sefr=0
      var pass_l=[]
      for (var i=0; i<5; i++){
        if (i+1===a_dict[step-1]["agent_1"][0]){
          pass_l.push(yek)
        }
        else{
          pass_l.push(sefr)
        }
        l_dict[step]["agent_1"].push(pass_l[i])
      }
      for (var i=0; i<5; i++){
        if (i+1===a_dict[step-1]["agent_2"][0]){
          pass_l.push(yek)
        }
        else{
          pass_l.push(sefr)
        }
        l_dict[step]["agent_2"].push(pass_l[i+5])
      }
      document.querySelector(".agent-1-last_action .l_1_a_1 p").innerText=pass_l[0].toString()
      document.querySelector(".agent-1-last_action .l_2_a_1 p").innerText=pass_l[1].toString()
      document.querySelector(".agent-1-last_action .l_3_a_1 p").innerText=pass_l[2].toString()
      document.querySelector(".agent-1-last_action .l_4_a_1 p").innerText=pass_l[3].toString()
      document.querySelector(".agent-1-last_action .l_5_a_1 p").innerText=pass_l[4].toString()
      document.querySelector(".agent-2-last_action .l_1_a_2 p").innerText=pass_l[5].toString()
      document.querySelector(".agent-2-last_action .l_2_a_2 p").innerText=pass_l[6].toString()
      document.querySelector(".agent-2-last_action .l_3_a_2 p").innerText=pass_l[7].toString()
      document.querySelector(".agent-2-last_action .l_4_a_2 p").innerText=pass_l[8].toString()
      document.querySelector(".agent-2-last_action .l_5_a_2 p").innerText=pass_l[9].toString()
      get_l_a=1
      on_off="on"
    }
  }
  else{
    disp(get_obs,get_l_a,get_h,get_input,get_q,get_l_s,get_state)
    document.querySelector(".agent-1-last_action").style.display="block"
    document.querySelector(".agent-2-last_action").style.display="block"
  }
})



// for getting hidden_state
document.querySelectorAll(".btn-apply-fun")[2].addEventListener("click",function(event){
  if(get_h==0){
    disp(get_obs,get_l_a,get_h,get_input,get_q,get_l_s)
    document.querySelector(".agent-1-hidden_dim").style.display="block"
    document.querySelector(".agent-2-hidden_dim").style.display="block"
    if (step===1){
      var pass_h=[]
      for (var i=0; i<4; i++){
        pass_h.push(sefr)
        h_dict[step]["agent_1"].push(pass_h[i])
      }
      for (var i=0; i<4; i++){
        pass_h.push(sefr)
        h_dict[step]["agent_2"].push(pass_h[i+4])
      }
      sefr=0
      document.querySelector(".agent-1-hidden_dim .h_1_a_1 p").innerText=pass_h[0].toString()
      document.querySelector(".agent-1-hidden_dim .h_2_a_1 p").innerText=pass_h[1].toString()
      document.querySelector(".agent-1-hidden_dim .h_3_a_1 p").innerText=pass_h[2].toString()
      document.querySelector(".agent-1-hidden_dim .h_4_a_1 p").innerText=pass_h[3].toString()
      document.querySelector(".agent-2-hidden_dim .h_1_a_2 p").innerText=pass_h[4].toString()
      document.querySelector(".agent-2-hidden_dim .h_2_a_2 p").innerText=pass_h[5].toString()
      document.querySelector(".agent-2-hidden_dim .h_3_a_2 p").innerText=pass_h[6].toString()
      document.querySelector(".agent-2-hidden_dim .h_4_a_2 p").innerText=pass_h[7].toString()
      get_h=1
      on_off="on"
    }
    else{
      //deficit
      document.querySelector(".agent-1-hidden_dim .h_1_a_1 p").innerText=hidden_dict[step-1]["agent_1"][0].toString()
      document.querySelector(".agent-1-hidden_dim .h_2_a_1 p").innerText=hidden_dict[step-1]["agent_1"][1].toString()
      document.querySelector(".agent-1-hidden_dim .h_3_a_1 p").innerText=hidden_dict[step-1]["agent_1"][2].toString()
      document.querySelector(".agent-1-hidden_dim .h_4_a_1 p").innerText=hidden_dict[step-1]["agent_1"][3].toString()
      document.querySelector(".agent-2-hidden_dim .h_1_a_2 p").innerText=hidden_dict[step-1]["agent_2"][0].toString()
      document.querySelector(".agent-2-hidden_dim .h_2_a_2 p").innerText=hidden_dict[step-1]["agent_2"][1].toString()
      document.querySelector(".agent-2-hidden_dim .h_3_a_2 p").innerText=hidden_dict[step-1]["agent_2"][2].toString()
      document.querySelector(".agent-2-hidden_dim .h_4_a_2 p").innerText=hidden_dict[step-1]["agent_2"][3].toString()
      get_h=1
      on_off="on"
    }
  }
  else{
    disp(get_obs,get_l_a,get_h,get_input,get_q,get_l_s,get_state)
    document.querySelector(".agent-1-hidden_dim").style.display="block"
    document.querySelector(".agent-2-hidden_dim").style.display="block"
  }
})


// for getting input_dim
document.querySelectorAll(".btn-apply-fun")[3].addEventListener("click",function(event){
  if (get_input==0){
    disp(get_obs,get_l_a,get_h,get_input,get_q,get_l_s,get_state)
    document.querySelector(".final-output-base").style.display="block"
    document.querySelector(".agent-1-input_dim .i_1_a_1 p").innerText=o_dict[step]["agent_1"][0].toString()
    document.querySelector(".agent-1-input_dim .i_2_a_1 p").innerText=o_dict[step]["agent_1"][1].toString()
    document.querySelector(".agent-1-input_dim .i_3_a_1 p").innerText=o_dict[step]["agent_1"][2].toString()
    document.querySelector(".agent-1-input_dim .i_4_a_1 p").innerText=o_dict[step]["agent_1"][3].toString()
    document.querySelector(".agent-1-input_dim .i_1_2_a_1 p").innerText=l_dict[step]["agent_1"][0].toString()
    document.querySelector(".agent-1-input_dim .i_2_2_a_1 p").innerText=l_dict[step]["agent_1"][1].toString()
    document.querySelector(".agent-1-input_dim .i_3_2_a_1 p").innerText=l_dict[step]["agent_1"][2].toString()
    document.querySelector(".agent-1-input_dim .i_4_2_a_1 p").innerText=l_dict[step]["agent_1"][3].toString()
    document.querySelector(".agent-1-input_dim .i_5_2_a_1 p").innerText=l_dict[step]["agent_1"][4].toString()
    document.querySelector(".agent-2-input_dim .i_1_a_2 p").innerText=o_dict[step]["agent_2"][0].toString()
    document.querySelector(".agent-2-input_dim .i_2_a_2 p").innerText=o_dict[step]["agent_2"][1].toString()
    document.querySelector(".agent-2-input_dim .i_3_a_2 p").innerText=o_dict[step]["agent_2"][2].toString()
    document.querySelector(".agent-2-input_dim .i_4_a_2 p").innerText=o_dict[step]["agent_2"][3].toString()
    document.querySelector(".agent-2-input_dim .i_1_2_a_2 p").innerText=l_dict[step]["agent_2"][0].toString()
    document.querySelector(".agent-2-input_dim .i_2_2_a_2 p").innerText=l_dict[step]["agent_2"][1].toString()
    document.querySelector(".agent-2-input_dim .i_3_2_a_2 p").innerText=l_dict[step]["agent_2"][2].toString()
    document.querySelector(".agent-2-input_dim .i_4_2_a_2 p").innerText=l_dict[step]["agent_2"][3].toString()
    document.querySelector(".agent-2-input_dim .i_5_2_a_2 p").innerText=l_dict[step]["agent_2"][4].toString()
    get_input=1
    on_off="on"
  }
  else{
    disp(get_obs,get_l_a,get_h,get_input,get_q,get_l_s,get_state)
    document.querySelector(".final-output-base").style.display="block"
  }
})


//get action
document.querySelectorAll(".btn-apply-fun")[4].addEventListener("click",function(event){
  if(get_q==0){
    disp(get_obs,get_l_a,get_h,get_input,get_q,get_l_s,get_state)
    document.querySelector(".get_action_base").style.display="flex"
    var pass_q=[]
    for (var i=0; i<5; i++){
      pass_q.push(2*Math.random().toFixed(1))
      q_dict[step]["agent_1"].push(pass_q[i])
    }
    for (var i=0; i<5; i++){
      pass_q.push(2*Math.random().toFixed(1))
      q_dict[step]["agent_2"].push(pass_q[i+5])
    }
    document.querySelector(".get_action_base .q_1_a_1 p").innerText=q_dict[step]["agent_1"][0].toString()
    document.querySelector(".get_action_base .q_2_a_1 p").innerText=q_dict[step]["agent_1"][1].toString()
    document.querySelector(".get_action_base .q_3_a_1 p").innerText=q_dict[step]["agent_1"][2].toString()
    document.querySelector(".get_action_base .q_4_a_1 p").innerText=q_dict[step]["agent_1"][3].toString()
    document.querySelector(".get_action_base .q_5_a_1 p").innerText=q_dict[step]["agent_1"][4].toString()
    document.querySelector(".get_action_base .q_1_a_2 p").innerText=q_dict[step]["agent_2"][0].toString()
    document.querySelector(".get_action_base .q_2_a_2 p").innerText=q_dict[step]["agent_2"][1].toString()
    document.querySelector(".get_action_base .q_3_a_2 p").innerText=q_dict[step]["agent_2"][2].toString()
    document.querySelector(".get_action_base .q_4_a_2 p").innerText=q_dict[step]["agent_2"][3].toString()
    document.querySelector(".get_action_base .q_5_a_2 p").innerText=q_dict[step]["agent_2"][4].toString()

    var pass_hidden=[]
    for (var i=0; i<4; i++){
      pass_hidden.push(2*Math.random().toFixed(1))
      hidden_dict[step]["agent_1"].push(pass_hidden[i])
    }
    for (var i=0; i<4; i++){
      pass_hidden.push(2*Math.random().toFixed(1))
      hidden_dict[step]["agent_2"].push(pass_hidden[i+4])
    }
    document.querySelector(".get_action_base .hidden_1_a_1 p").innerText=hidden_dict[step]["agent_1"][0].toString()
    document.querySelector(".get_action_base .hidden_2_a_1 p").innerText=hidden_dict[step]["agent_1"][1].toString()
    document.querySelector(".get_action_base .hidden_3_a_1 p").innerText=hidden_dict[step]["agent_1"][2].toString()
    document.querySelector(".get_action_base .hidden_4_a_1 p").innerText=hidden_dict[step]["agent_1"][3].toString()
    document.querySelector(".get_action_base .hidden_1_a_2 p").innerText=hidden_dict[step]["agent_2"][0].toString()
    document.querySelector(".get_action_base .hidden_2_a_2 p").innerText=hidden_dict[step]["agent_2"][1].toString()
    document.querySelector(".get_action_base .hidden_3_a_2 p").innerText=hidden_dict[step]["agent_2"][2].toString()
    document.querySelector(".get_action_base .hidden_4_a_2 p").innerText=hidden_dict[step]["agent_2"][3].toString()




    a_dict[step]["agent_1"].push(Math.floor(Math.random()*5)+1)
    a_dict[step]["agent_2"].push(Math.floor(Math.random()*5)+1)
    get_q=1
    on_off="on"
  }
  else{
    disp(get_obs,get_l_a,get_h,get_input,get_q,get_l_s,get_state)
    document.querySelector(".get_action_base").style.display="flex"
  }
})


//get last_settings
document.querySelectorAll(".btn-apply-fun")[5].addEventListener("click",function(event){
  if (get_l_s==0){
    disp(get_obs,get_l_a,get_h,get_input,get_q,get_l_s,get_state)
    document.querySelector(".last_settings").style.display="flex"
    document.querySelectorAll(".last_settings .col .agents-reward p")[0].innerText="agent_1 reward is :"+Math.floor(2*Math.random()).toString()
    document.querySelectorAll(".last_settings .col .agents-reward p")[1].innerText="agent_2 reward is :"+Math.floor(2*Math.random()).toString()
    document.querySelectorAll(".last_settings .col .agents-terminated p")[0].innerText="agent_1 terminated is :"+Math.floor(0*Math.random()).toString()
    document.querySelectorAll(".last_settings .col .agents-terminated p")[1].innerText="agent_2 terminated is :"+Math.floor(0*Math.random()).toString()
    document.querySelectorAll(".last_settings .col .agents-padded p")[0].innerText="agent_1 padded is :"+Math.floor(0*Math.random()).toString()
    document.querySelectorAll(".last_settings .col .agents-padded p")[1].innerText="agent_2 padded is :"+Math.floor(0*Math.random()).toString()
    pass_1=a_dict[step]["agent_1"][0]
    pass_2=a_dict[step]["agent_2"][0]
    var one_hot_a_1=""
    var one_hot_a_2=""
    for (var i=1; i<6; i++){
      if (pass_1===i){
        one_hot_a_1=one_hot_a_1+"1"
      }
      else{
        one_hot_a_1=one_hot_a_1+"0"
      }
    }
    for (var i=1; i<6; i++){
      if (pass_2===i){
        one_hot_a_2=one_hot_a_2+"1"
      }
      else{
        one_hot_a_2=one_hot_a_2+"0"
      }
    }

    document.querySelectorAll(".last_settings .col .agents-one-hot p")[0].innerText="agent_1 action-one-hot is :"+one_hot_a_1
    document.querySelectorAll(".last_settings .col .agents-one-hot p")[1].innerText="agent_2 action-one-hot is :"+one_hot_a_2


    for (var i=0; i<4; i++){
      o_dict[step+1]["agent_1"].push(2*Math.random().toFixed(1))
    }
    for (var i=0; i<4; i++){
      o_dict[step+1]["agent_2"].push(2*Math.random().toFixed(1))
    }
    document.querySelector(".agents-new-obs-1 .agent-1-new-obs .o_1_a_1 p").innerText=(o_dict[step+1]["agent_1"][0]).toString()
    document.querySelector(".agents-new-obs-1 .agent-1-new-obs .o_2_a_1 p").innerText=(o_dict[step+1]["agent_1"][1]).toString()
    document.querySelector(".agents-new-obs-1 .agent-1-new-obs .o_3_a_1 p").innerText=(o_dict[step+1]["agent_1"][2]).toString()
    document.querySelector(".agents-new-obs-1 .agent-1-new-obs .o_4_a_1 p").innerText=(o_dict[step+1]["agent_1"][3]).toString()
    document.querySelector(".agents-new-obs-2 .agent-2-new-obs .o_1_a_2 p").innerText=(o_dict[step+1]["agent_2"][0]).toString()
    document.querySelector(".agents-new-obs-2 .agent-2-new-obs .o_2_a_2 p").innerText=(o_dict[step+1]["agent_2"][1]).toString()
    document.querySelector(".agents-new-obs-2 .agent-2-new-obs .o_3_a_2 p").innerText=(o_dict[step+1]["agent_2"][2]).toString()
    document.querySelector(".agents-new-obs-2 .agent-2-new-obs .o_4_a_2 p").innerText=(o_dict[step+1]["agent_2"][3]).toString()

    on_off="off"
    get_l_s=1
  }
  else{
    disp(get_obs,get_l_a,get_h,get_input,get_q,get_l_s,get_state)
    document.querySelector(".last_settings").style.display="flex"
  }
})



//go go_next_step
document.querySelector(".go_next_step").addEventListener("click",function(event){
  document.querySelector(".last_settings").style.display="none"
  step+=1
  document.querySelector(".step-show .btn").innerText="step "+step.toString()+" run"

  passive_variable=document.querySelectorAll(".btn-apply-fun")
  for (var i=0; i<passive_variable.length; i++){
    passive_variable[i].style.display="none"
  }

  get_obs=0
  get_l_a=0
  get_h=0
  get_input=0
  get_q=0
  get_l_s=0
  count_fun=1


  if (step===5){
    document.querySelector(".step-show").style.display="none"
    document.querySelector(".step-show-n").style.display="none"
      document.querySelector(".reset-env-class").style.bottom="-5px"
    document.querySelector(".obs-title").style.display="block"
    document.querySelector(".buffer-place").style.display="block"
    document.querySelector(".discript-steps").style.display="block"
    document.querySelector(".go_next_stored_vars").style.display="block"
  }


})



// for going to next function
document.querySelector(".step-show-n .btn").addEventListener("click",function(event){
  if (passive_variable!==0){
    if (on_off=="on"){
      if (count_fun==1){
        passive_variable[1].style.display="block"
        if (get_obs==1){
          document.querySelector(".agent-1-obs").style.display="none"
          document.querySelector(".agent-2-obs").style.display="none"
          document.querySelector(".agent-1-state").style.display="none"
          document.querySelector(".agent-2-state").style.display="none"
          document.querySelector(".agent-state-total").style.display="none"
          on_off="off"
        }
      }
      if (count_fun==2){
        passive_variable[2].style.display="block"
        if (get_l_a==1){
          document.querySelector(".agent-1-last_action").style.display="none"
          document.querySelector(".agent-2-last_action").style.display="none"
          on_off="off"
        }
      }
      if (count_fun==3){
        passive_variable[3].style.display="block"
        if (get_h==1){
          document.querySelector(".agent-1-hidden_dim").style.display="none"
          document.querySelector(".agent-2-hidden_dim").style.display="none"
          on_off="off"
        }
      }
      if (count_fun==4){
        passive_variable[4].style.display="block"
        if (get_input===1){
          document.querySelector(".final-output-base").style.display="none"
          on_off="off"
        }

      }
      if (count_fun==5){
        passive_variable[5].style.display="block"
        if (get_q===1){
          document.querySelector(".get_action_base").style.display="none"
          on_off="off"
        }
        count_fun=count_fun+1
      }
      count_fun=count_fun+1
    }
  }
})



//go next stored various

var stored_o=true
var stored_o_n=false
var stored_s=false
var stored_s_n=false
var stored_r=false
var stored_t=false
var stored_p=false
var stored_u=false
var stored_u_o=false
var stored_a_a_c=false
var stored_a_a_n=false

document.querySelector(".go_next_stored_vars").addEventListener("click",function(event){
  if(stored_o===true){
    document.querySelector(".obs-title p").innerText="Observation_next"
    document.querySelector(".discript-steps").innerHTML="after episode terminating, observation from step " + "<strong style=" + "font-family:initial;" + ">2</strong> : <strong style=" + "font-family:initial;" + ">episode_limit</strong> are stored in buffer"
    var p=document.querySelectorAll(".buffer-place-1 p")
    for (var i=0; i<p.length; i++){
      p[i].innerText="step_"+ (i+2).toString() + " (1x78)"
      if (i===2){
        p[i].innerText="to"
      }
      if (i===3){
        p[i].innerText="step ("+ "episode_limit) " + " (1x78)"
      }
    }
    var p=document.querySelectorAll(".buffer-place-2 p")
    for (var i=0; i<p.length; i++){
      p[i].innerText="step_"+ (i+2).toString() + " (1x78)"
      if (i===2){
        p[i].innerText="to"
      }
      if (i===3){
        p[i].innerText="step ("+ "episode_limit) " + " (1x78)"
      }
    }
    stored_o=false
    stored_o_n=true
  }
  else if(stored_o_n===true){
    document.querySelector(".obs-title p").innerText="State"
    document.querySelector(".discript-steps").innerHTML="after episode terminating, states from step " + "<strong style=" + "font-family:initial;" + ">1</strong> : <strong style=" + "font-family:initial;" + ">episode_limit-1</strong> are stored in buffer"
    document.querySelector(".buffer-place-1").style.display="none"
    var p=document.querySelectorAll(".buffer-place-2 p")
    for (var i=0; i<p.length; i++){
      p[i].innerText="step_"+ (i+1).toString() + " (1x142)"
      if (i===2){
        p[i].innerText="to"
      }
      if (i===3){
        p[i].innerText="step ("+ "episode_limit-1) " + " (1x142)"
      }
    }
    stored_o_n=false
    stored_s=true
  }
  else if(stored_s===true){
    document.querySelector(".obs-title p").innerText="State-next"
    document.querySelector(".discript-steps").innerHTML="after episode terminating, states from step " + "<strong style=" + "font-family:initial;" + ">2</strong> : <strong style=" + "font-family:initial;" + ">episode_limit</strong> are stored in buffer"
    var p=document.querySelectorAll(".buffer-place-2 p")
    for (var i=0; i<p.length; i++){
      p[i].innerText="step_"+ (i+2).toString() + " (1x142)"
      if (i===2){
        p[i].innerText="to"
      }
      if (i===3){
        p[i].innerText="step ("+ "episode_limit) " + " (1x142)"
      }
    }
    stored_s=false
    stored_s_n=true
  }
  else if(stored_s_n===true){
    document.querySelector(".obs-title p").innerText="rewards"
    document.querySelector(".discript-steps").innerHTML="after episode terminating, rewards from step " + "<strong style=" + "font-family:initial;" + ">1</strong> : <strong style=" + "font-family:initial;" + ">episode_limit</strong> are stored in buffer"
    var p=document.querySelectorAll(".buffer-place-2 p")
    for (var i=0; i<p.length; i++){
      p[i].innerText="step_"+ (i+1).toString() + " (1x1)"
      if (i===2){
        p[i].innerText="to"
      }
      if (i===3){
        p[i].innerText="step ("+ "episode_limit) " + " (1x1)"
      }
    }
    stored_s_n=false
    stored_r=true
  }
  else if(stored_r===true){
    document.querySelector(".obs-title p").innerText="terminated"
    document.querySelector(".discript-steps").innerHTML="after episode terminating, terminated values from step " + "<strong style=" + "font-family:initial;" + ">1</strong> : <strong style=" + "font-family:initial;" + ">episode_limit</strong> are stored in buffer"
    var p=document.querySelectorAll(".buffer-place-2 p")
    for (var i=0; i<p.length; i++){
      p[i].innerText="step_"+ (i+1).toString() + " (1x1)"
      if (i===2){
        p[i].innerText="to"
      }
      if (i===3){
        p[i].innerText="step ("+ "episode_limit) " + " (1x1)"
      }
    }
    stored_r=false
    stored_t=true
  }
  else if(stored_t===true){
    document.querySelector(".obs-title p").innerText="padded"
    document.querySelector(".discript-steps").innerHTML="after episode terminating, padded values from step " + "<strong style=" + "font-family:initial;" + ">1</strong> : <strong style=" + "font-family:initial;" + ">episode_limit</strong> are stored in buffer"
    var p=document.querySelectorAll(".buffer-place-2 p")
    for (var i=0; i<p.length; i++){
      p[i].innerText="step_"+ (i+1).toString() + " (1x1)"
      if (i===2){
        p[i].innerText="to"
      }
      if (i===3){
        p[i].innerText="step ("+ "episode_limit) " + " (1x1)"
      }
    }
    stored_t=false
    stored_p=true
  }
  else if(stored_p===true){
    document.querySelector(".obs-title p").innerText="actions"
    document.querySelector(".discript-steps").innerHTML="after episode terminating, action values from step " + "<strong style=" + "font-family:initial;" + ">1</strong> : <strong style=" + "font-family:initial;" + ">episode_limit</strong> are stored in buffer"
    var p=document.querySelectorAll(".buffer-place-2 p")
    for (var i=0; i<p.length; i++){
      p[i].innerText="step_"+ (i+1).toString() + " (1x2)"
      if (i===2){
        p[i].innerText="to"
      }
      if (i===3){
        p[i].innerText="step ("+ "episode_limit) " + " (1x2)"
      }
    }
    stored_p=false
    stored_u=true
  }
  else if(stored_u===true){
    document.querySelector(".obs-title p").innerText="actions_one_hot"
    document.querySelector(".discript-steps").innerHTML="after episode terminating, action (one_hot) values from step " + "<strong style=" + "font-family:initial;" + ">1</strong> : <strong style=" + "font-family:initial;" + ">episode_limit</strong> are stored in buffer"
    var p=document.querySelectorAll(".buffer-place-2 p")
    for (var i=0; i<p.length; i++){
      p[i].innerText="step_"+ (i+1).toString() + " (2x5)"
      if (i===2){
        p[i].innerText="to"
      }
      if (i===3){
        p[i].innerText="step ("+ "episode_limit) " + " (2x5)"
      }
    }
    stored_u=false
    stored_u_o=true
  }
  else if(stored_u_o===true){
    document.querySelector(".obs-title p").innerText="actions_available"
    document.querySelector(".discript-steps").innerHTML="after episode terminating, action (available) values from step " + "<strong style=" + "font-family:initial;" + ">1</strong> : <strong style=" + "font-family:initial;" + ">episode_limit-1</strong> are stored in buffer"
    var p=document.querySelectorAll(".buffer-place-2 p")
    for (var i=0; i<p.length; i++){
      p[i].innerText="step_"+ (i+1).toString() + " (2x5)"
      if (i===2){
        p[i].innerText="to"
      }
      if (i===3){
        p[i].innerText="step ("+ "episode_limit-1) " + " (2x5)"
      }
    }
    stored_u_o=false
    stored_a_a_c=true
  }
  else if(stored_a_a_c===true){
    document.querySelector(".obs-title p").innerText="actions_available_next"
    document.querySelector(".discript-steps").innerHTML="after episode terminating, action (one_hot) values from step " + "<strong style=" + "font-family:initial;" + ">2</strong> : <strong style=" + "font-family:initial;" + ">episode_limit</strong> are stored in buffer"
    var p=document.querySelectorAll(".buffer-place-2 p")
    for (var i=0; i<p.length; i++){
      p[i].innerText="step_"+ (i+2).toString() + " (2x5)"
      if (i===2){
        p[i].innerText="to"
      }
      if (i===3){
        p[i].innerText="step ("+ "episode_limit) " + " (2x5)"
      }
    }
    stored_a_a_c=false
    stored_a_a_n=true
  }
})



// reset
document.querySelector(".reset_env").addEventListener("click",function(event){

  disp(get_obs,get_l_a,get_h,get_input,get_q,get_l_s,get_state)

  passive_variable=document.querySelectorAll(".btn-apply-fun")
  for (var i=0; i<passive_variable.length; i++){
    passive_variable[i].style.display="none"
  }


  count_fun=1

  on_off="off"

  get_obs=0

  get_l_a=0

  get_h=0

  get_input=0

  get_q=0

  get_l_s=0

  step=1

  document.querySelector(".obs-title").style.display="none"
  document.querySelector(".buffer-place").style.display="none"
  document.querySelector(".discript-steps").style.display="none"
  document.querySelector(".go_next_stored_vars").style.display="none"


  document.querySelector(".step-show").style.display="inline-block"
  document.querySelector(".step-show-n").style.display="inline-block"
  document.querySelector(".reset-env-class").style.bottom="55px"


  document.querySelector(".step-show .btn").innerText="step "+step.toString()+" run"
})


function disp(get_obs,get_l_a,get_h,get_input,get_q,get_l_s,get_state){
  if (get_obs==1){
    document.querySelector(".agent-1-obs").style.display="none"
    document.querySelector(".agent-2-obs").style.display="none"
  }
  if (get_l_a==1){
    document.querySelector(".agent-1-last_action").style.display="none"
    document.querySelector(".agent-2-last_action").style.display="none"
  }
  if (get_h==1){
    document.querySelector(".agent-1-hidden_dim").style.display="none"
    document.querySelector(".agent-2-hidden_dim").style.display="none"
  }
  if (get_input==1){
  document.querySelector(".final-output-base").style.display="none"
  }
  if (get_q==1){
  document.querySelector(".get_action_base").style.display="none"
  }
  if (get_l_s==1){
      document.querySelector(".last_settings").style.display="none"
  }
  if (get_state==1){
    document.querySelector(".agent-1-state").style.display="none"
    document.querySelector(".agent-2-state").style.display="none"
    document.querySelector(".agent-state-total").style.display="none"
  }
}

















// train part


var n_agents=2

var episode_limit=5

var obs_shape=71

var obs_shape_final=78

var state_shape=142

var n_actions=5

var batch_size="none"

var active_take_sample=0

var active_get_key=0

var active_feed_inputs=0

var active_feed_hypernetwork=0

var active_backward=0

var get_take_sample=0

var get_feed_inputs=0

var get_feed_hypernetwork=0

var get_bacward=0

var gamma=0.01




// begin take sample
document.querySelector(".btn-take-sample").addEventListener("click",function(event){
  if (active_feed_inputs==1){
    document.querySelector(".feed-inputs-last-part").style.display="none"
  }
  if(active_feed_hypernetwork==1){
    active_feed_hypernetwork=0
    document.querySelector(".feed-hypernetwork").style.display="none"
  }
  if(active_backward==1){
    active_backward=0
    document.querySelector(".backward-process").style.display="none"
  }
  active_take_sample=1
  document.querySelector(".take_sample_buffer_base_base").style.display="block"
})


// (get) key
document.querySelector("form .btn").addEventListener("click",function(event){
  batch_size=document.querySelector("#fname").value
  if (batch_size!=="" && Number.isInteger(parseInt(batch_size))){
    active_get_key=1
    get_take_sample=1
    document.querySelector(".after-get-process").style.display="block"
    document.querySelector(".after-get-process .col-lg p").innerText="take " + batch_size + " sample"
    var batch_explain=document.querySelectorAll(".batch_explain")
    batch_explain[1].innerText="obs (" + batch_size + "x" + episode_limit.toString() + "x" + n_agents.toString() + "x" + obs_shape_final.toString() + " )"
    batch_explain[2].innerText="obs_next (" + batch_size + "x" + episode_limit.toString() + "x" + n_agents.toString() + "x" + obs_shape_final.toString() + " )"
    batch_explain[3].innerText="state (" + batch_size + "x" + episode_limit.toString() + "x" + "1" + "x" + state_shape.toString() + " )"
    batch_explain[4].innerText="state_next (" + batch_size + "x" + episode_limit.toString() + "x" + "1" + "x" + state_shape.toString() + " )"
    batch_explain[5].innerText="actions (" + batch_size + "x" + episode_limit.toString() + "x" + n_agents.toString() + "x" + "1" + " )"
    batch_explain[6].innerText="rewards (" + batch_size + "x" + episode_limit.toString() + "x" + "1" + " )"
    batch_explain[7].innerText="terminated (" + batch_size + "x" + episode_limit.toString() + "x" +  "1" + " )"
    batch_explain[8].innerText="padded (" + batch_size + "x" + episode_limit.toString() + "x" + "1" + " )"
    batch_explain[9].innerText="action_onehot (" + batch_size + "x" + episode_limit.toString() + "x" + n_agents.toString() + "x" + n_actions.toString() + " )"
  }
})


// manipulate top_part and below_part_1
document.querySelector(".btn-feed-inputs").addEventListener("click",function(event){
  if (active_take_sample==1){
    active_take_sample=0
    document.querySelector(".take_sample_buffer_base_base").style.display="none"
  }
  if (active_get_key==1){
    active_get_key=0
    document.querySelector(".after-get-process").style.display="none"
  }
  if(active_feed_hypernetwork==1){
    active_feed_hypernetwork=0
    document.querySelector(".feed-hypernetwork").style.display="none"
  }
  if(active_backward==1){
    active_backward=0
    document.querySelector(".backward-process").style.display="none"
  }
  active_feed_inputs=1
  if(get_take_sample==1){
    document.querySelector(".feed-inputs-last-part").style.display="block"
  }
  if (Number.isInteger(parseInt(batch_size))==true && get_take_sample==1){
    get_feed_inputs=1
    var take_p_1=document.querySelectorAll(".feed-inputs-last-part .top_part_1 p")
    take_p_1[0].innerText="Q (" + batch_size + "x" + episode_limit.toString() + "x" + n_agents.toString() + "x" + n_actions + ")"
    take_p_1[2].innerText="wrt --> u( " + batch_size + "x" + episode_limit.toString() + "x"  + n_agents.toString() + "x" +  "1) "
    take_p_1[4].innerText="Q( " + batch_size + "x" + episode_limit.toString() + "x"  + n_agents.toString() + "x" +  "1) "

    var take_p_2=document.querySelectorAll(".feed-inputs-last-part .top_part_2 p")
    take_p_2[0].innerText="Q_target (" + batch_size + "x" + episode_limit.toString() + "x" + n_agents.toString() + "x" + n_actions + ")"
    take_p_2[2].innerText="take maximum"
    take_p_2[4].innerText="Q_target( " + batch_size + "x" + episode_limit.toString() + "x"  + n_agents.toString() + "x" +  "1) "

    document.querySelector(".below_part_1").innerText="observation batch from buffer --> ( " + batch_size + "x"  + episode_limit.toString() + "x" + n_agents.toString() + "x" + "78" + ")"

    document.querySelector(".below_part_2").innerText="observation_next batch from buffer --> ( " + batch_size + "x"  + episode_limit.toString() + "x" + n_agents.toString() + "x" + "78" + ")"

  }

})


// feed hypernetwork
document.querySelector(".btn-hypernetwork").addEventListener('click',function(event){
  if (active_take_sample==1){
    active_get_key=0
    document.querySelector(".take_sample_buffer_base_base").style.display="none"
  }
  if (active_get_key==1){
    active_get_key=0
    document.querySelector(".after-get-process").style.display="none"
  }
  if(active_feed_inputs==1){
    active_feed_inputs=0
    document.querySelector(".feed-inputs-last-part").style.display="none"
  }
  if(active_backward==1){
    active_backward=0
    document.querySelector(".backward-process").style.display="none"
  }
  active_feed_hypernetwork=1
  if (get_take_sample==1 && get_feed_inputs==1){
    document.querySelector(".feed-hypernetwork").style.display="block"
  }
  if (Number.isInteger(parseInt(batch_size))==true && get_take_sample==1 && get_feed_inputs==1){
    get_feed_hypernetwork=1
    var feed_states_1=document.querySelectorAll(".states-second-place")
    var feed_states_2=document.querySelectorAll(".states-initial-place")
    var weight_places=document.querySelectorAll(".hypernet-weight-place")
    var q_final_place=document.querySelector(".q-final-place")
    var q_place=document.querySelectorAll(".q-place")
    var w_place=document.querySelectorAll(".w-place")
    var batch_ep=parseInt(batch_size)*episode_limit
    feed_states_1[0].innerText="states("+ batch_ep.toString() +"x"+ state_shape.toString()+")"
    feed_states_1[1].innerText="states("+ batch_ep.toString() +"x"+ state_shape.toString()+")"
    feed_states_1[2].innerText="states("+ batch_ep.toString() +"x"+ state_shape.toString()+")"
    feed_states_1[3].innerText="states("+ batch_ep.toString() +"x"+ state_shape.toString()+")"

    feed_states_2[0].innerText="states("+ batch_size +"x"+ episode_limit.toString() + "x" + state_shape.toString()+")"
    feed_states_2[1].innerText="states("+ batch_size +"x"+ episode_limit.toString() + "x" + state_shape.toString()+")"
    feed_states_2[2].innerText="states("+ batch_size +"x"+ episode_limit.toString() + "x" + state_shape.toString()+")"

    weight_places[0].innerText="hypernet layer_2 bias (1 node)"
    weight_places[0].style.height="40px"
    weight_places[1].innerText="hypernet layer_2 bias (32 node)"
    weight_places[1].style.height="40px"
    weight_places[2].innerText="hypernet layer_2 weight (32 node)"
    weight_places[2].style.height="40px"
    weight_places[3].innerText="hypernet layer_1 bias (32 node)"
    weight_places[3].style.height="40px"
    weight_places[4].innerText="hypernet layer_1 weight (2*32 node)"
    weight_places[4].style.height="40px"

    q_final_place.innerText="q_total (" + n_agents.toString() + "x" + episode_limit.toString() + "x" + "1)"

    q_place[0].innerText="q_values(" + batch_size + "x" + episode_limit.toString() + "x" + "1" + "x" + n_agents.toString() + ")"
    q_place[1].innerText="q_values(" + batch_size + "x" + episode_limit.toString() + "x" + n_agents.toString() + "x" + "1)"

    w_place[1].style.height="40px"

    document.querySelector(".feed-hypernetwork").style.fontSize="11px"
  }
})

document.querySelector(".go-next-feed-hyper").addEventListener("click",function(event){
  var q_final_place=document.querySelector(".q-final-place")
  var q_place=document.querySelectorAll(".q-place")

  q_final_place.innerText="q_total_target (" + n_agents.toString() + "x" + episode_limit.toString() + "x" + "1)"

  q_place[0].innerText="q_targets(" + batch_size + "x" + episode_limit.toString() + "x" + "1" + "x" + n_agents.toString() + ")"
  q_place[1].innerText="q_targets(" + batch_size + "x" + episode_limit.toString() + "x" + n_agents.toString() + "x" + "1)"

  q_final_place.style.backgroundColor="cornflowerblue"

  q_place[0].style.backgroundColor="cornflowerblue"
  q_place[1].style.backgroundColor="cornflowerblue"
})

document.querySelector(".go-back-feed-hyper").addEventListener("click",function(event){
  var q_final_place=document.querySelector(".q-final-place")
  var q_place=document.querySelectorAll(".q-place")

  q_final_place.innerText="q_total (" + n_agents.toString() + "x" + episode_limit.toString() + "x" + "1)"

  q_place[0].innerText="q_values(" + batch_size + "x" + episode_limit.toString() + "x" + "1" + "x" + n_agents.toString() + ")"
  q_place[1].innerText="q_values(" + batch_size + "x" + episode_limit.toString() + "x" + n_agents.toString() + "x" + "1)"

  q_final_place.style.backgroundColor="darkorange"

  q_place[0].style.backgroundColor="darkorange"
  q_place[1].style.backgroundColor="darkorange"
})



document.querySelector(".btn-backward").addEventListener("click",function(event){
  if (active_take_sample==1){
    active_get_key=0
    document.querySelector(".take_sample_buffer_base_base").style.display="none"
  }
  if (active_get_key==1){
    active_get_key=0
    document.querySelector(".after-get-process").style.display="none"
  }
  if(active_feed_inputs==1){
    active_feed_inputs=0
    document.querySelector(".feed-inputs-last-part").style.display="none"
  }
  if(active_feed_hypernetwork==1){
    active_feed_hypernetwork=0
    document.querySelector(".feed-hypernetwork").style.display="none"
  }
  active_backward=1
  if (get_take_sample==1 && get_feed_inputs==1 && get_feed_hypernetwork==1){
    document.querySelector(".backward-process").style.display="block"
  }
  if (Number.isInteger(parseInt(batch_size))==true && get_take_sample==1 && get_feed_inputs==1){
    get_bacward=1
    document.querySelector(".loss-backward-1").innerText="Target = " + "Reward" + "(" + batch_size + "x" + episode_limit.toString() + "x" + "1" + ")" + " + " + gamma.toString() +  "*Q_target" + "(" + batch_size + "x" + episode_limit.toString() + "x" + "1" + ")" + "*(1-terminated" + "(" + batch_size + "x" + episode_limit.toString() + "x" + "1" + ")" + ")"
    document.querySelector(".loss-backward-2").innerText="loss=sum( (Q_total-Target)**2 )  --> dloss"
  }
})
