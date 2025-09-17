#!/usr/bin/env node

/**
 * Azure AI IT Copilot - Stripe Product Setup
 * Creates products and pricing in your Stripe account
 */

const stripe = require('stripe');

// Load credentials from secure location
require('dotenv').config({
    path: '/home/mookyjooky/Dropbox/SIDEGIGS/05 - Scripts & Tools/SECURE-DO-NOT-COMMIT/stripe-live-config.env'
});

// Initialize Stripe with your live key
const stripeClient = stripe(process.env.STRIPE_SECRET_KEY);

// Product configuration for Azure AI IT Copilot
const PRODUCT_CONFIG = {
    name: 'Azure AI IT Copilot',
    description: 'AI-Powered IT Operations Platform for Azure',
    metadata: {
        category: 'saas',
        type: 'it-operations',
        ai_powered: 'true'
    }
};

// Pricing tiers
const PRICING_TIERS = [
    {
        nickname: 'Starter',
        unit_amount: 29900, // $299 in cents
        currency: 'usd',
        recurring: {
            interval: 'month'
        },
        metadata: {
            tier: 'starter',
            max_resources: '50',
            max_api_calls: '1000',
            agents: '5',
            support: 'email'
        }
    },
    {
        nickname: 'Professional',
        unit_amount: 99900, // $999 in cents
        currency: 'usd',
        recurring: {
            interval: 'month'
        },
        metadata: {
            tier: 'professional',
            max_resources: '500',
            max_api_calls: '10000',
            agents: 'all',
            support: 'priority'
        }
    },
    {
        nickname: 'Enterprise',
        unit_amount: 299900, // $2999 in cents
        currency: 'usd',
        recurring: {
            interval: 'month'
        },
        metadata: {
            tier: 'enterprise',
            max_resources: 'unlimited',
            max_api_calls: '100000',
            agents: 'all+custom',
            support: 'dedicated'
        }
    }
];

// Usage-based pricing for API calls
const USAGE_PRICING = {
    nickname: 'API Calls Overage',
    currency: 'usd',
    recurring: {
        interval: 'month',
        usage_type: 'metered'
    },
    tiers: [
        {
            up_to: 1000,
            unit_amount: 0 // Free tier
        },
        {
            up_to: 10000,
            unit_amount: 10 // $0.10 per call
        },
        {
            up_to: 'inf',
            unit_amount: 5 // $0.05 per call after 10k
        }
    ],
    tiers_mode: 'graduated'
};

async function createProduct() {
    try {
        console.log('🚀 Creating Azure AI IT Copilot product in Stripe...\n');

        // Create the main product
        const product = await stripeClient.products.create(PRODUCT_CONFIG);
        console.log('✅ Product created:', product.id);
        console.log('   Name:', product.name);
        console.log('');

        // Create pricing for each tier
        console.log('💰 Creating pricing tiers...\n');

        const priceIds = {};
        for (const tier of PRICING_TIERS) {
            const price = await stripeClient.prices.create({
                product: product.id,
                ...tier
            });

            priceIds[tier.nickname.toLowerCase()] = price.id;
            console.log(`✅ ${tier.nickname} tier created:`);
            console.log(`   Price ID: ${price.id}`);
            console.log(`   Amount: $${tier.unit_amount / 100}/month`);
            console.log('');
        }

        // Create metered pricing for API overage
        console.log('📊 Creating usage-based pricing...\n');
        const usagePrice = await stripeClient.prices.create({
            product: product.id,
            ...USAGE_PRICING
        });
        console.log('✅ Usage pricing created:', usagePrice.id);
        console.log('');

        // Create checkout links for each tier
        console.log('🔗 Creating checkout links...\n');

        for (const [tierName, priceId] of Object.entries(priceIds)) {
            const session = await stripeClient.checkout.sessions.create({
                line_items: [{
                    price: priceId,
                    quantity: 1,
                }],
                mode: 'subscription',
                success_url: 'https://azureaiitcopilot.com/welcome?session_id={CHECKOUT_SESSION_ID}',
                cancel_url: 'https://azureaiitcopilot.com/pricing',
                allow_promotion_codes: true,
                billing_address_collection: 'required',
                customer_creation: 'always',
                payment_method_collection: 'always',
                metadata: {
                    product: 'azure_ai_copilot',
                    tier: tierName
                }
            });

            console.log(`✅ ${tierName.charAt(0).toUpperCase() + tierName.slice(1)} checkout:`);
            console.log(`   ${session.url}`);
            console.log('');
        }

        // Output environment variables for the application
        console.log('📝 Add these to your .env.production file:\n');
        console.log('```');
        console.log(`STRIPE_PRODUCT_ID=${product.id}`);
        console.log(`STRIPE_STARTER_PRICE_ID=${priceIds.starter}`);
        console.log(`STRIPE_PROFESSIONAL_PRICE_ID=${priceIds.professional}`);
        console.log(`STRIPE_ENTERPRISE_PRICE_ID=${priceIds.enterprise}`);
        console.log(`STRIPE_USAGE_PRICE_ID=${usagePrice.id}`);
        console.log('```');
        console.log('');

        // Create webhook endpoint
        console.log('🔔 Setting up webhook endpoint...\n');

        const webhook = await stripeClient.webhookEndpoints.create({
            url: 'https://azureaiitcopilot.com/api/stripe/webhook',
            enabled_events: [
                'checkout.session.completed',
                'customer.subscription.created',
                'customer.subscription.updated',
                'customer.subscription.deleted',
                'invoice.payment_succeeded',
                'invoice.payment_failed',
                'customer.updated',
                'payment_method.attached'
            ],
            description: 'Azure AI IT Copilot webhook'
        });

        console.log('✅ Webhook endpoint created:');
        console.log(`   URL: ${webhook.url}`);
        console.log(`   Secret: ${webhook.secret}`);
        console.log('');
        console.log('Add this to your .env.production:');
        console.log(`STRIPE_WEBHOOK_SECRET=${webhook.secret}`);
        console.log('');

        // Create a test customer
        console.log('👤 Creating test customer...\n');

        const customer = await stripeClient.customers.create({
            email: 'test@azureaiitcopilot.com',
            name: 'Test Customer',
            description: 'Test customer for Azure AI IT Copilot',
            metadata: {
                environment: 'test'
            }
        });

        console.log('✅ Test customer created:', customer.id);
        console.log('');

        // Summary
        console.log('=' .repeat(60));
        console.log('🎉 Stripe setup complete!\n');
        console.log('Next steps:');
        console.log('1. Update DNS to point azureaiitcopilot.com to your server');
        console.log('2. Deploy the application using deploy-to-digitalocean.sh');
        console.log('3. Test the checkout flows with the links above');
        console.log('4. Configure webhook endpoint in production');
        console.log('5. Enable live mode when ready');
        console.log('=' .repeat(60));

    } catch (error) {
        console.error('❌ Error setting up Stripe:', error.message);
        if (error.raw) {
            console.error('   Details:', error.raw.message);
        }
        process.exit(1);
    }
}

// Run if executed directly
if (require.main === module) {
    console.log('=' .repeat(60));
    console.log('Azure AI IT Copilot - Stripe Product Setup');
    console.log('=' .repeat(60));
    console.log('');

    // Check for Stripe key
    if (!process.env.STRIPE_SECRET_KEY) {
        console.error('❌ Stripe secret key not found!');
        console.error('   Please ensure stripe-live-config.env exists');
        process.exit(1);
    }

    // Confirm with user
    const readline = require('readline').createInterface({
        input: process.stdin,
        output: process.stdout
    });

    readline.question('This will create LIVE products in Stripe. Continue? (yes/no): ', (answer) => {
        if (answer.toLowerCase() === 'yes') {
            readline.close();
            createProduct();
        } else {
            console.log('Cancelled.');
            readline.close();
        }
    });
}

module.exports = { createProduct };