"use client"
import { useState, ChangeEvent, useEffect, SetStateAction } from "react";
import { useRouter } from "next/router";

const Search=()=>{
    const [search, setSearch] = useState('');
    const [history, setHistory] = useState<String []>([]);
    const [historyOn, setHistoryOn] = useState(false);

    //This function get the value(string) from the local storage, and add into the searchhistory list.
    const addSearchHistory=()=>{  
        const a = JSON.parse(localStorage.getItem("riotID")||"[]");
        setHistory(a)
    }

    const historyList=()=>{
        console.log(history)
        console.log(!history)
        if (history.length === 0) return <div>No history</div>
        else return <ul>{history.map((name, index) => <li key={index}>{name}</li>)}</ul>
    }
    //adding entered value into localstorage
    const handleSubmit = (value: String) => {
        setHistory([...history, value])
    }

    useEffect(() => {
        addSearchHistory();
    },[])

    useEffect (() =>{
        localStorage.setItem("riotID",JSON.stringify(history))
    }, [history])

    //This is a function for triggering to search by press enter key
    const handleKeyPress = (e) => {
        console.log(e)
        if (e.key == 'Enter'){
            console.log("perfect")
            console.log(e.key)
            handleSubmit(e.target.value)
        }
    }

    return(
        <div>
        <input 
        onFocus={()=> setHistoryOn(true)}
        onBlur={()=> setHistoryOn(false)}
        onChange={(e)=>setSearch(e.target.value)}
        onKeyDown={(e)=>{handleKeyPress(e);}}>
        
        </input>
        <div>{historyOn && historyList()}</div>
        </div>
    )
}

export default Search;