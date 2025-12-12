document.addEventListener('DOMContentLoaded', () => {

    const ChatNormativo = {
        // Elementos da UI
        elements: {
            chatContainer: document.getElementById('chat-container'),
            messageInput: document.getElementById('message-input'),
            sendButton: document.getElementById('send-button'),
            newChatButton: document.getElementById('new-chat-button'),
            typingIndicatorContainer: document.getElementById('typing-indicator'),
            examplePrompts: document.getElementById('example-prompts'),
            dropdownButton: document.querySelector('.dropbtn'),
            dbSwitch: document.getElementById('db-switch'),
            headerTitle: document.getElementById('header-title'),
            infoText: document.getElementById('info-text'),
        },

        // Estado do Chat
        state: {
            sessionId: null,
            isLoading: false,
            currentDb: 'cnj', // Valor padrão
        },

        // Endereço da API
        API_URL: '/chatnormas/legislacao',

        // Configurações de tema
        themes: {
            cnj: {
                title: 'CNJ',
                infoText: 'Converse com um agente de IA treinado em atos normativos do CNJ. Atualmente, o catálogo possui mais de 5800 atos normativos do CNJ.',
                welcomeMessage: 'Olá! Sou seu assistente virtual especializado em atos normativos do CNJ. Como posso ajudar você hoje?',
            },
            tjap: {
                title: 'TJAP',
                infoText: 'Converse com um agente de IA treinado exclusivamente em atos normativos internos do Tribunal de Justiça do Amapá (TJAP). Atualmente, o catálogo possui mais de 2500 atos internos.',
                welcomeMessage: 'Olá! Sou seu assistente virtual especializado em atos normativos do TJAP. Como posso ajudar você hoje?',
            }
        },

        /**
         * Inicializa o módulo do chat
         */
        init() {
            console.log("=== Chat ChatNormas - INÍCIO ===");
            
            // Garante que todos os elementos essenciais foram encontrados
            if (!this.elements.chatContainer || !this.elements.messageInput || 
                !this.elements.sendButton || !this.elements.typingIndicatorContainer) {
                console.error("ERRO CRÍTICO: Um ou mais elementos essenciais do chat não foram encontrados no HTML.");
                return;
            }
            
            // Carrega preferência salva do banco de dados
            const savedDb = localStorage.getItem('chatDb');
            if (savedDb === 'tjap') {
                this.state.currentDb = 'tjap';
                this.elements.dbSwitch.checked = true;
                this.applyTheme('tjap');
            } else {
                this.state.currentDb = 'cnj';
                this.applyTheme('cnj');
            }
            
            this.state.sessionId = this.getSessionId();
            this.bindEvents();
            this.loadChatHistory();
            
            // Adiciona mensagem de boas-vindas se o histórico estiver vazio
            if (this.elements.chatContainer.children.length <= 1) {
                this.elements.chatContainer.innerHTML = '';
                const welcomeMsg = this.themes[this.state.currentDb].welcomeMessage;
                this.addMessage(welcomeMsg, 'bot');
            }
            
            this.scrollToBottom();
            console.log("Session ID:", this.state.sessionId);
            console.log("Database:", this.state.currentDb);
        },

        /**
         * Aplica o tema visual baseado no banco de dados selecionado
         */
        applyTheme(db) {
            const theme = this.themes[db];
            
            // Atualiza classes do body
            if (db === 'tjap') {
                document.body.classList.add('theme-tjap');
            } else {
                document.body.classList.remove('theme-tjap');
            }
            
            // Atualiza título
            this.elements.headerTitle.textContent = theme.title;
            
            // Atualiza texto informativo
            this.elements.infoText.textContent = theme.infoText;
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
            
            // Evento do switch
            this.elements.dbSwitch.addEventListener('change', (e) => {
                const newDb = e.target.checked ? 'tjap' : 'cnj';
                this.state.currentDb = newDb;
                localStorage.setItem('chatDb', newDb);
                this.applyTheme(newDb);
                
                // Reinicia o chat ao trocar de banco
                this.resetChat(false); // false = não gera novo sessionId
                
                console.log("Banco de dados alterado para:", newDb);
            });
            
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
                    body: JSON.stringify({ 
                        message: userInput, 
                        sessionId: this.state.sessionId,
                        db: this.state.currentDb // Envia o banco selecionado
                    })
                });

                if (!response.ok) {
                    const errorData = await response.json().catch(() => ({ 
                        detail: 'Resposta de erro inválida do servidor.' 
                    }));
                    throw new Error(errorData.detail || `Erro na API: ${response.status}`);
                }

                const data = await response.json();
                const botReply = data.output || "Não recebi uma resposta válida do servidor.";
                this.addMessage(botReply, 'bot');

            } catch (error) {
                console.error("ERRO em sendMessage:", error);
                this.addMessage(
                    `Desculpe, ocorreu um erro de comunicação com a IA. (${error.message})`, 
                    'bot', 
                    true
                );
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
        resetChat(generateNewSession = true) {
            console.log("A reiniciar o chat.");
            localStorage.removeItem(`chatHistory_${this.state.sessionId}_${this.state.currentDb}`);
            
            if (generateNewSession) {
                this.state.sessionId = this.getSessionId(true);
            }
            
            this.elements.chatContainer.innerHTML = '';
            const welcomeMsg = this.themes[this.state.currentDb].welcomeMessage;
            this.addMessage(welcomeMsg, 'bot');
            
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
            setTimeout(() => {
                this.elements.chatContainer.scrollTop = this.elements.chatContainer.scrollHeight;
            }, 50);
        },

        saveChatHistory() {
            const history = Array.from(this.elements.chatContainer.children).map(msg => ({
                role: msg.classList.contains('user-message') ? 'user' : 'bot',
                content: msg.querySelector('.message-content').innerHTML
            }));
            const storageKey = `chatHistory_${this.state.sessionId}_${this.state.currentDb}`;
            localStorage.setItem(storageKey, JSON.stringify(history));
        },

        loadChatHistory() {
            const storageKey = `chatHistory_${this.state.sessionId}_${this.state.currentDb}`;
            const savedHistory = localStorage.getItem(storageKey);
            
            if (savedHistory) {
                this.elements.chatContainer.innerHTML = ''; 
                const history = JSON.parse(savedHistory);
                if (history.length === 0) return;
                
                history.forEach(msg => {
                    const messageElement = this.createMessageElement('', msg.role);
                    messageElement.querySelector('.message-content').innerHTML = msg.content;
                    this.elements.chatContainer.appendChild(messageElement);
                });
            }
        },

        parseMarkdown(text) {
            if (typeof text !== 'string') return '';
            return marked.parse(text, { gfm: true, breaks: true });
        }
    };

    ChatNormativo.init();
});