document.addEventListener('DOMContentLoaded', () => {

    const ChatNormativo = {
        // Elementos da UI
        elements: {
            chatContainer: document.getElementById('chat-container'),
            messageInput: document.getElementById('message-input'),
            sendButton: document.getElementById('send-button'),
            newChatButton: document.getElementById('new-chat-button'),
            // CORREÇÃO: O ID no HTML é 'typing-indicator', não 'typing-indicator-container'
            typingIndicatorContainer: document.getElementById('typing-indicator'),
            examplePrompts: document.getElementById('example-prompts'),
            dropdownButton: document.querySelector('.dropbtn'),
        },

        // Estado do Chat
        state: {
            sessionId: null,
            isLoading: false,
        },

        // Endereço da API (conforme seu routes.py)
        API_URL: '/chatnormas/legislacao',

        /**
         * Inicializa o módulo do chat
         */
        init() {
            console.log("=== Chat ChatNormas - INÍCIO ===");
            // Garante que todos os elementos essenciais foram encontrados
            if (!this.elements.chatContainer || !this.elements.messageInput || !this.elements.sendButton || !this.elements.typingIndicatorContainer) {
                console.error("ERRO CRÍTICO: Um ou mais elementos essenciais do chat não foram encontrados no HTML. Verifique os IDs.");
                return;
            }
            this.state.sessionId = this.getSessionId();
            this.bindEvents();
            this.loadChatHistory();
            // Adiciona mensagem de boas-vindas se o histórico estiver vazio
            if (this.elements.chatContainer.children.length <= 1) { // Apenas a mensagem inicial do HTML
                 this.elements.chatContainer.innerHTML = ''; // Limpa para não duplicar
                 this.addMessage("Olá! Sou seu assistente virtual especializado em propostas de governo. Como posso ajudar você hoje?", 'bot');
            }
            this.scrollToBottom();
            console.log("Session ID:", this.state.sessionId);
        },

        /**
         * Associa todos os eventos da UI
         */
        bindEvents() {
            this.elements.sendButton.addEventListener('click', () => this.sendMessage());
            this.elements.messageInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.sendMessage();
                }
            });
            this.elements.newChatButton.addEventListener('click', () => this.resetChat());
            
            this.elements.dropdownButton.addEventListener('click', (e) => {
                e.stopPropagation();
                this.elements.examplePrompts.classList.toggle('show');
            });
            
            this.elements.examplePrompts.addEventListener('click', (e) => {
                if (e.target.tagName === 'A' && e.target.dataset.prompt) {
                    e.preventDefault();
                    this.elements.messageInput.value = e.target.dataset.prompt;
                    this.sendMessage();
                    this.elements.examplePrompts.classList.remove('show');
                }
            });

            window.addEventListener('click', (e) => {
                if (!e.target.matches('.dropbtn, .dropbtn *')) {
                    if (this.elements.examplePrompts.classList.contains('show')) {
                        this.elements.examplePrompts.classList.remove('show');
                    }
                }
            });
        },

        /**
         * Obtém ou cria um ID de sessão único
         * @param {boolean} forceNew - Força a criação de um novo ID de sessão
         */
        getSessionId(forceNew = false) {
            let sessionId = localStorage.getItem('chatSessionId');
            if (forceNew || !sessionId) {
                sessionId = `session_${Date.now()}_${Math.random().toString(36).substring(2, 11)}`;
                localStorage.setItem('chatSessionId', sessionId);
            }
            return sessionId;
        },

        /**
         * Função principal para enviar a mensagem do usuário
         */
        async sendMessage() {
            const userInput = this.elements.messageInput.value.trim();
            if (!userInput || this.state.isLoading) return;

            this.state.isLoading = true;
            this.addMessage(userInput, 'user');
            this.elements.messageInput.value = '';
            this.elements.messageInput.focus();
            this.showTypingIndicator();

            try {
                const response = await fetch(this.API_URL, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: userInput, sessionId: this.state.sessionId })
                });

                if (!response.ok) {
                    const errorData = await response.json().catch(() => ({ detail: 'Resposta de erro inválida do servidor.' }));
                    throw new Error(errorData.detail || `Erro na API: ${response.status}`);
                }

                const data = await response.json();
                // A sua API retorna um objeto com a chave "output" dentro
                const botReply = data.output || "Não recebi uma resposta válida do servidor.";
                this.addMessage(botReply, 'bot');

            } catch (error) {
                console.error("ERRO em sendMessage:", error);
                this.addMessage(`Desculpe, ocorreu um erro de comunicação com a IA. (${error.message})`, 'bot', true);
            } finally {
                this.state.isLoading = false;
                this.hideTypingIndicator();
            }
        },

        /**
         * Adiciona uma mensagem à UI e ao histórico
         */
        addMessage(content, role, isError = false) {
            const messageElement = this.createMessageElement(content, role, isError);
            this.elements.chatContainer.appendChild(messageElement);
            this.saveChatHistory();
            this.scrollToBottom();
        },

        /**
         * Cria o elemento DOM para uma mensagem
         */
        createMessageElement(content, role, isError) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${role}-message`;
            if (isError) messageDiv.classList.add('error-message');

            const avatarHTML = `
                <div class="avatar">
                    ${role === 'bot' ? 
                    '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22a10 10 0 0 0-3.95-7.95l-1.05-1.05A10 10 0 0 1 12 2Z"></path><path d="M12 22a10 10 0 0 1 3.95-7.95l1.05-1.05A10 10 0 0 0 12 2Z"></path><path d="m9 14 3-3 3 3"></path><path d="M9 14v1"></path><path d="M15 14v1"></path><path d="M12 11v6"></path></svg>' : 
                    '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>'}
                </div>`;
            
            const contentHTML = `<div class="message-content">${this.parseMarkdown(content)}</div>`;

            messageDiv.innerHTML = avatarHTML + contentHTML;
            return messageDiv;
        },

        /**
         * Reinicia o chat para um novo estado
         */
        resetChat() {
            console.log("A reiniciar o chat.");
            localStorage.removeItem(`chatHistory_${this.state.sessionId}`);
            this.state.sessionId = this.getSessionId(true); // Força nova sessão
            this.elements.chatContainer.innerHTML = '';
            this.addMessage("Olá! Sou seu assistente virtual especializado em propostas de governo. Como posso ajudar você hoje?", 'bot');
            console.log("Nova Session ID:", this.state.sessionId);
        },
        
        showTypingIndicator() {
            this.elements.typingIndicatorContainer.innerHTML = `
                <div class="typing-indicator">
                    <div class="dot"></div><div class="dot"></div><div class="dot"></div>
                </div>`;
            this.scrollToBottom();
        },

        hideTypingIndicator() {
            this.elements.typingIndicatorContainer.innerHTML = '';
        },

        scrollToBottom() {
            // Adicionado um pequeno delay para garantir que o DOM foi atualizado
            setTimeout(() => {
                this.elements.chatContainer.scrollTop = this.elements.chatContainer.scrollHeight;
            }, 50);
        },

        saveChatHistory() {
            const history = Array.from(this.elements.chatContainer.children).map(msg => ({
                role: msg.classList.contains('user-message') ? 'user' : 'bot',
                content: msg.querySelector('.message-content p').innerHTML
            }));
            localStorage.setItem(`chatHistory_${this.state.sessionId}`, JSON.stringify(history));
        },

        loadChatHistory() {
            const savedHistory = localStorage.getItem(`chatHistory_${this.state.sessionId}`);
            if (savedHistory) {
                this.elements.chatContainer.innerHTML = ''; 
                const history = JSON.parse(savedHistory);
                if (history.length === 0) return; // Não carrega nada se o histórico salvo estiver vazio
                history.forEach(msg => {
                    const messageElement = this.createMessageElement('', msg.role);
                    messageElement.querySelector('.message-content p').innerHTML = msg.content;
                    this.elements.chatContainer.appendChild(messageElement);
                });
            }
        },

        parseMarkdown(text) {
            if (typeof text !== 'string') return '';

            // A mágica acontece aqui!
            // A biblioteca 'marked' converte TUDO (títulos, listas, negrito, etc.) para HTML.
            // O 'gfm: true' habilita a compatibilidade com o markdown do GitHub (mais comum).
            return marked.parse(text, { gfm: true, breaks: true });
        }
     };

    ChatNormativo.init();
});