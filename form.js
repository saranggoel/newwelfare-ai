// Your web app's Firebase configuration
var firebaseConfig = {
    apiKey: "AIzaSyCOBK9LhFfTLcZHJduCHm3ekGib32NK54Q",
    authDomain: "welfare-ai-2bb0d.firebaseapp.com",
    projectId: "welfare-ai-2bb0d",
    storageBucket: "welfare-ai-2bb0d.appspot.com",
    messagingSenderId: "1097265132697",
    appId: "1:1097265132697:web:90cd1813a1ec96b297b408"
};
// Initialize Firebase
firebase.initializeApp(firebaseConfig);

const auth = firebase.auth();

function signUp() {
    var email = document.getElementById("email");
    var password = document.getElementById("password");
    const promise = auth.createUserWithEmailAndPassword(email.value, password.value)
    promise.catch(e => alert(e.message))
    alert("Signed Up")
}

function signIn() {
    var email = document.getElementById("email");
    var password = document.getElementById("password");
    const promise = auth.signInWithEmailAndPassword(email.value, password.value)
    promise.catch(e => alert(e.message))
    alert("Signed In")
    window.location.pathname = "D:/newwelfare-ai-main/newwelfare-ai-main/inner-page.html";
}

function signOut() {
    auth.signOut();
    alert("Signed Out")
}

