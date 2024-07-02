css = '''
<style>
    .stTextInput {
        position: fixed;
        bottom: 0; /* Fix the chat bar at the bottom of the screen */
        
        background-color: #fff; /* Set the background color for the chat bar */
        padding: 1rem; /* Add some padding for better visual appearance */
        box-sizing: border-box; /* Include padding and border in the element's total width and height */
        border-top: 1px solid #ccc; /* Add a border at the top for separation */
    }

.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://dalleproduse.blob.core.windows.net/private/images/9d66951c-2fd2-4926-bb83-54e00f8e76df/generated_00.png?se=2023-08-24T20%3A23%3A45Z&sig=kdi3sI0722LKoueXFUbDq3WjMhPvZx7JSZPvF%2FRPO7Y%3D&ske=2023-08-30T15%3A38%3A41Z&skoid=09ba021e-c417-441c-b203-c81e5dcd7b7f&sks=b&skt=2023-08-23T15%3A38%3A41Z&sktid=33e01921-4d64-4f8c-a055-5bdaffd5e33d&skv=2020-10-02&sp=r&spr=https&sr=b&sv=2020-10-02">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://dalleproduse.blob.core.windows.net/private/images/1e1414f2-d604-40fc-b92f-f5042247dd2a/generated_00.png?se=2023-08-23T20%3A48%3A58Z&sig=NnrMgCkQFs9%2Fewy60meMdMREZGmdKjiMI0DpZHEgIwg%3D&ske=2023-08-29T14%3A45%3A41Z&skoid=09ba021e-c417-441c-b203-c81e5dcd7b7f&sks=b&skt=2023-08-22T14%3A45%3A41Z&sktid=33e01921-4d64-4f8c-a055-5bdaffd5e33d&skv=2020-10-02&sp=r&spr=https&sr=b&sv=2020-10-02">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''
